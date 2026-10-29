"""Export a fixed FP32 Android candidate and verify numerical parity."""
import argparse
import os
from pathlib import Path

os.environ['MAMBA_SCAN_BACKEND'] = 'torch'
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.nn.modules import common_utils_mbyolo as scan_utils


class ExportCrossScan:
    @staticmethod
    def apply(x):
        rows = x.flatten(2)
        cols = x.transpose(2, 3).flatten(2)
        return torch.stack((rows, cols, rows.flip(-1), cols.flip(-1)), dim=1)


class ExportCrossMerge:
    @staticmethod
    def apply(x):
        b, k, d, h, w = x.shape
        x = x.reshape(b, k, d, h * w)
        rows = x[:, 0] + x[:, 2].flip(-1)
        cols = x[:, 1] + x[:, 3].flip(-1)
        return rows + cols.reshape(b, d, w, h).transpose(2, 3).reshape(b, d, h * w)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = YOLO(args.weights).model.cpu().float().eval()
    head = model.model[-1]
    head.export = True
    head.format = 'onnx'
    head.dynamic = False
    out = Path(__file__).parent / 'exports' / 'android'
    out.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        reference = model(x).numpy()
        assert np.isfinite(reference).all(), 'Invalid PyTorch output'
        scan_utils.CrossScan = ExportCrossScan
        scan_utils.CrossMerge = ExportCrossMerge
        np.testing.assert_allclose(model(x).numpy(), reference, rtol=1e-4, atol=1e-4)
        torch.onnx.export(model, x, str(out / 'model.onnx'), opset_version=17,
                          input_names=['images'], output_names=['predictions'])
    onnx.checker.check_model(str(out / 'model.onnx'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(out / 'model.onnx'), options,
                                   providers=['CPUExecutionProvider'])
    actual = session.run(None, {'images': x.numpy()})[0]
    np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-3)
    print('PASS: ONNX matches PyTorch', actual.shape, flush=True)
