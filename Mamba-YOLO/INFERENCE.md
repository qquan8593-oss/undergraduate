# 拉取后进行图片预测（Windows）

需要 Python 3.11 和网络连接以安装依赖。无需下载 VisDrone 数据集，也无需编译 CUDA 算子。推荐先使用 CPU 跑通。

## 1. 打开项目目录

克隆本仓库后，在 PowerShell 中进入包含 `predict_images.py` 的 `Mamba-YOLO` 目录。

## 2. 建立独立环境并安装依赖

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
```

不需要激活环境。若没有 `py` 命令，用自己的 Python 3.11 解释器完整路径代替第一行的 `py -3.11`。
不要另行安装 PyPI 上的 ultralytics 替代本仓库，它不包含这里的 Mamba 模块。

## 3. 预测图片

把下面的路径换成你自己的图片路径：

```powershell
.\.venv\Scripts\python.exe predict_images.py --source "C:\images\test.jpg" --device cpu
```

`--source` 也支持图片文件夹。结果默认保存在 `output_dir/predict_last`，文件名以 `_pred.jpg` 结尾；重复运行同名图片会覆盖对应结果。

默认使用仓库内 `weights/visdrone_last.pt`，无需修改代码或提供原作者电脑上的路径。默认图片大小为 256、置信度阈值为 0.25，强制 FP32。

```powershell
# 查看更多低置信度候选框，误检也会增加
.\.venv\Scripts\python.exe predict_images.py --source "C:\images\test.jpg" --device cpu --conf 0.05
```

## 可选 NVIDIA GPU

在同一虚拟环境中，用以下命令替换 CPU 版 PyTorch（需要兼容的 NVIDIA 驱动）：

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe predict_images.py --source "C:\images\test.jpg" --device 0
```

## 权重说明

附带权重来自 VisDrone 的 5 轮训练（256 输入、batch=4），用于演示推理，不是作者官方预训练权重。已经观察到 FP16 推理数值异常，因此脚本使用 FP32。它可以产生预测框，但存在误检、漏检，完整有效精度尚未确认。

运行时出现 pure PyTorch selective scan 警告是预期行为。这是兼容实现，CPU/GPU 均可使用，但不能用于复现论文速度。

维护者推送前请确认 `weights/visdrone_last.pt` 已包含在提交中；其他训练权重、数据集和输出仍按忽略规则处理。
