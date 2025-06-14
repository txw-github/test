
@echo off
chcp 65001
echo ========================================
echo 中文电视剧音频转文字工具 - RTX 3060 Ti专版
echo ========================================

echo 第一步：检查系统环境...
echo ========================================

echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python！
    echo 请从 https://www.python.org/downloads/ 下载Python 3.10
    echo 安装时请勾选"Add Python to PATH"
    pause
    exit /b 1
)

python --version
echo Python环境检查通过✓

echo.
echo 检查NVIDIA驱动...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到NVIDIA驱动！
    echo 请从NVIDIA官网下载最新驱动程序
    pause
    exit /b 1
) else (
    echo NVIDIA驱动检查通过✓
    nvidia-smi
)

echo.
echo 检查CUDA版本...
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')" 2>nul
if errorlevel 1 (
    echo CUDA检查将在安装PyTorch后进行
) else (
    echo CUDA环境检查通过✓
)

echo.
echo 第二步：创建工作目录...
echo ========================================
if not exist "models" mkdir models
echo 已创建models目录用于存储模型文件

echo.
echo 第三步：安装依赖包...
echo ========================================

echo 升级pip...
python -m pip install --upgrade pip

echo.
echo 安装PyTorch CUDA 12.1版本（这可能需要几分钟）...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo PyTorch安装失败，请检查网络连接
    pause
    exit /b 1
)

echo.
echo 验证CUDA安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

echo.
echo 安装Whisper相关库...
pip install openai-whisper faster-whisper

echo.
echo 安装音频处理库...
pip install librosa soundfile scipy numpy

echo.
echo 安装视频处理库...
pip install moviepy

echo.
echo 安装其他必需库...
pip install tqdm psutil transformers accelerate

echo.
echo 安装ModelScope（FireRedASR支持）...
pip install modelscope datasets==2.14.0

echo.
echo 尝试安装TensorRT加速（可选）...
pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com
if errorlevel 1 (
    echo TensorRT安装失败，将跳过加速功能（不影响基本使用）
)

pip install pycuda
if errorlevel 1 (
    echo PyCuda安装失败，将跳过TensorRT加速
)

echo.
echo 第四步：测试安装结果...
echo ========================================

echo 测试核心功能...
python test_installation.py

echo.
echo ========================================
echo 安装完成！
echo ========================================

echo 使用说明：
echo 1. 将视频文件放在此目录下
echo 2. 双击运行 start_conversion.bat
echo 3. 按提示选择模型和输入文件名

pause
