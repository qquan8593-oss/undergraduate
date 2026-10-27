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

## 在 VisDrone 上从零训练

训练入口默认使用 `ultralytics/cfg/datasets/VisDrone.yaml`，包含 VisDrone2019-DET 的 10 个类别。
模型从结构 YAML 创建，不加载 COCO 预训练权重。检测头由训练器根据数据集自动调整为 10 类，通用模型结构文件中的 `nc: 80` 不需要手动修改。
默认输入 256×256、batch=1、workers=0，优先使用 GPU；默认训练轮数仍为 300，可通过 `--epochs` 修改。

数据默认放在 `C:/Users/111/Desktop/本科项目/Mamba-YOLO/datasets/VisDrone`。
首次运行训练时，如果缺少验证集图片，会自动下载训练集和验证集并转换 YOLO 标签；不会下载 COCO 或竞赛测试集。
如果使用已有数据，请修改 VisDrone.yaml 的 `path`，并确保 train/val 子目录下都有 images 和转换后的 labels。只有图片目录已存在时，框架不会自动执行转换脚本。

先运行一轮检查真实数据的耗时和显存：

```powershell
& 'C:\Users\111\AppData\Local\Programs\Python\Python311\python.exe' mbyolo_train.py --epochs 1
```

确认数据、显存和损失正常后，例如训练 100 轮：

```powershell
& 'C:\Users\111\AppData\Local\Programs\Python\Python311\python.exe' mbyolo_train.py --epochs 100
```

结果保存在 `output_dir/visdrone/mambayolo_t_visdrone*`，训练完成后查看其中的 `weights/best.pt`。
256 是本机兼容实现的起步尺寸，不是最终小目标实验效果的保证；提高分辨率前需重新测试显存。
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
