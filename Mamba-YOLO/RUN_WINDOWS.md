# Windows / Python 3.11.5 运行说明

解释器：`C:\Users\111\AppData\Local\Programs\Python\Python311\python.exe`。

当前采用 GPU 版 PyTorch 2.3.0+cu121、torchvision 0.18.0+cu121、torchaudio 2.3.0+cu121，以及纯 PyTorch selective scan 兼容实现。
不需要额外编译 CUDA 扩展。兼容实现支持自动求导，但速度和内存占用不等同于原版 CUDA 算子，不能用于论文速度对比。
模型结构和状态更新公式保持不变；并行前缀计算可能有浮点舍入差异。

## 验证环境

在 PowerShell 中进入本目录，运行：

```powershell
& 'C:\Users\111\AppData\Local\Programs\Python\Python311\python.exe' -m pip check
```

这会检查已安装依赖是否冲突。本次安装已验证扫描运算的输出及梯度，以及随机初始化的 Mamba YOLO-T 前向和反向传播。
随机权重验证仅证明代码执行正常，不代表有可用的检测精度。一次性验证脚本和日志已清理。

## 训练

先下载数据集并准备对应的 YOLO 标注和 data.yaml。数据文件中建议使用绝对路径。
确认小尺寸和小批量能运行后，再根据内存情况提高输入尺寸。

```powershell
& 'C:\Users\111\AppData\Local\Programs\Python\Python311\python.exe' mbyolo_train.py --task train --data '你的数据集配置.yaml' --device 0 --batch_size 1 --workers 0 --imgsz 128 --epochs 1
```

128 是初次排查环境用的输入尺寸，不是建议的小目标检测研究配置。正式训练需选择适合实验的数据与尺寸。
原项目默认 COCO 数据尚未下载，不能直接运行默认训练命令。
项目附带的 `yolov8n.pt` 也不是 Mamba YOLO 的训练权重。

## 在另一台机器重建环境

```powershell
& 'C:\Users\111\AppData\Local\Programs\Python\Python311\python.exe' -m pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
& 'C:\Users\111\AppData\Local\Programs\Python\Python311\python.exe' -m pip install -r requirements-windows-py311.txt
& 'C:\Users\111\AppData\Local\Programs\Python\Python311\python.exe' -m pip install --no-deps --no-build-isolation -e .
```

## 使用 GPU

本机 GTX 1660 Ti 有 6GB 显存，现已安装 CUDA 12.1 版 PyTorch。使用 `--device 0` 选择 GPU，启动脚本也会自动优先选择可用的 GPU。
纯 PyTorch 兼容算子可在 GPU 上执行，无需 C++ 编译器。建议先使用 batch=1 和小输入检查显存，再调整正式实验设置。
若要使用项目原版高性能 CUDA 算子，还需要另外安装合适的 CUDA Toolkit / C++ 编译器并完成 Windows 编译适配。

加载原生 CUDA 扩展成功且输入在 GPU 上时，代码会使用原版算子。
可设置环境变量 `MAMBA_SCAN_BACKEND=torch` 强制启用兼容实现。
