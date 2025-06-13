
@echo off
echo ========================================
echo 视频转字幕工具 - Windows安装脚本
echo ========================================

echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo 检查pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到pip
    pause
    exit /b 1
)

echo 检查CUDA环境...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo 警告：未检测到NVIDIA驱动，将使用CPU模式
) else (
    echo CUDA环境检测成功
)

echo.
echo 开始安装依赖包...
echo ========================================

echo 升级pip...
python -m pip install --upgrade pip

echo 安装PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo 安装基础依赖...
pip install numpy scipy librosa soundfile

echo 安装Whisper相关...
pip install openai-whisper faster-whisper

echo 安装Transformers...
pip install transformers accelerate

echo 安装视频处理...
pip install moviepy

echo 安装其他依赖...
pip install tqdm psutil jieba cn2an

echo 安装ModelScope (FireRedASR)...
pip install modelscope

echo 尝试安装TensorRT (可选)...
pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com

echo 安装PyCuda (可选)...
pip install pycuda

echo 安装音频处理...
pip install pyloudnorm

echo.
echo ========================================
echo 安装完成！
echo ========================================

echo 测试安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

echo.
echo 使用说明：
echo python main.py 你的视频文件.mp4 --model base --output 输出.srt
echo.
echo 支持的模型：
echo - tiny, base, small, medium, large (原版Whisper)
echo - faster-base, faster-large (Faster-Whisper)
echo - firered-aed, firered-llm (FireRedASR)

pause
