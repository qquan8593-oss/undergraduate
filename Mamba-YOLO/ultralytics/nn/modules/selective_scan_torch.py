"""Differentiable reference selective scan, without compiled extensions.

Uses a chunked associative prefix scan for x[t] = a[t] * x[t-1] + b[t].
No division by cumulative products is used, avoiding underflow instability.
This compatibility implementation prioritizes portability over throughput.
"""
import torch
import torch.nn.functional as F


def selective_scan_torch(u, delta, A, B, C, D=None, delta_bias=None,
                         delta_softplus=True, chunk_size=64):
    original_dtype = u.dtype
    u, delta, A, B, C = (v.float() for v in (u, delta, A, B, C))
    batch, channels, length = u.shape
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    def coefficients(v, start, end):
        if v.ndim == 2:  # Constant coefficients: (channels, states).
            return v.unsqueeze(0).unsqueeze(-1)
        if v.ndim == 3:
            v = v.unsqueeze(1)
        groups = v.shape[1]
        if channels % groups:
            raise ValueError("Selective scan channels must be divisible by coefficient groups")
        return v[..., start:end].repeat_interleave(channels // groups, dim=1)

    state = u.new_zeros(batch, channels, A.shape[-1])
    outputs = []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        dt = delta[..., start:end].unsqueeze(2)
        transition = torch.exp(dt * A.unsqueeze(0).unsqueeze(-1))
        drive = dt * coefficients(B, start, end) * u[..., start:end].unsqueeze(2)
        # Inclusive prefix composition of affine state transitions.
        step = 1
        while step < end - start:
            drive = torch.cat((drive[..., :step], drive[..., step:]
                               + transition[..., step:] * drive[..., :-step]), dim=-1)
            transition = torch.cat((transition[..., :step], transition[..., step:]
                                    * transition[..., :-step]), dim=-1)
            step *= 2
        states = drive + transition * state.unsqueeze(-1)
        outputs.append((states * coefficients(C, start, end)).sum(dim=2))
        state = states[..., -1]
    y = torch.cat(outputs, dim=-1)
    if D is not None:
        y = y + u * D.float().view(1, -1, 1)
    return y.to(original_dtype)
