
@echo off
echo ========================================
echo 视频转字幕工具 - RTX 3060 Ti 优化版安装
echo ========================================

echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python，请先安装Python 3.10
    pause
    exit /b 1
)

echo 检查NVIDIA驱动...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到NVIDIA驱动，请先安装显卡驱动
    pause
    exit /b 1
) else (
    echo NVIDIA驱动检测成功
)

echo 检查FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到FFmpeg，请先安装FFmpeg
    pause
    exit /b 1
) else (
    echo FFmpeg检测成功
)

echo.
echo 开始安装依赖包...
echo ========================================

echo 升级pip...
python -m pip install --upgrade pip

echo 安装PyTorch CUDA 12.1版本...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo 安装Whisper系列...
pip install openai-whisper
pip install faster-whisper

echo 安装音频处理库...
pip install librosa soundfile scipy numpy

echo 安装视频处理库...
pip install moviepy

echo 安装其他必需库...
pip install tqdm psutil

echo 安装HuggingFace相关...
pip install transformers accelerate

echo 降级datasets避免版本冲突...
pip install datasets==2.14.0

echo 尝试安装TensorRT（可选，如果失败可忽略）...
pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com
if errorlevel 1 (
    echo TensorRT安装失败，将跳过（不影响基本功能）
)

echo 尝试安装PyCuda（TensorRT加速需要）...
pip install pycuda
if errorlevel 1 (
    echo PyCuda安装失败，将跳过TensorRT加速
)

echo.
echo ========================================
echo 依赖安装完成！
echo ========================================

echo 测试PyTorch CUDA支持...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无GPU\"}')"

echo.
echo 测试Whisper...
python -c "import whisper; print('Whisper导入成功')"

echo 测试Faster-Whisper...
python -c "from faster_whisper import WhisperModel; print('Faster-Whisper导入成功')"

echo.
echo 安装完成！现在可以使用工具了
echo 使用示例：python main.py 你的视频.mp4 --model faster-base
echo.
pause
