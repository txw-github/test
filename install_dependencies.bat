
@echo off
chcp 65001
echo ===============================================================
echo RTX 3060 Ti 中文视频转字幕工具 - 自动安装脚本
echo ===============================================================
echo.

:: 检查Python版本
echo [1/8] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    echo 请从 https://www.python.org 下载并安装Python 3.8-3.11
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python版本: %PYTHON_VERSION%

:: 检查CUDA
echo [2/8] 检查CUDA环境...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  NVIDIA GPU驱动未检测到
    echo 建议安装最新NVIDIA驱动以获得最佳性能
) else (
    echo ✅ NVIDIA GPU驱动已安装
)

:: 升级pip
echo [3/8] 升级pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ❌ pip升级失败
    pause
    exit /b 1
)

:: 安装PyTorch (CUDA 12.1)
echo [4/8] 安装PyTorch (CUDA 12.1)...
echo 这可能需要几分钟时间，请耐心等待...
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ❌ PyTorch安装失败，尝试CPU版本...
    python -m pip install torch torchvision torchaudio
    if errorlevel 1 (
        echo ❌ PyTorch安装完全失败
        pause
        exit /b 1
    )
)

:: 安装Whisper模型
echo [5/8] 安装Whisper模型...
python -m pip install openai-whisper
python -m pip install faster-whisper
if errorlevel 1 (
    echo ❌ Whisper模型安装失败
    pause
    exit /b 1
)

:: 安装音视频处理库
echo [6/8] 安装音视频处理库...
python -m pip install moviepy librosa soundfile scipy
python -m pip install pydub ffmpeg-python
if errorlevel 1 (
    echo ❌ 音视频处理库安装失败
    pause
    exit /b 1
)

:: 安装中文处理库
echo [7/8] 安装中文处理库...
python -m pip install jieba zhon cn2an
python -m pip install transformers accelerate
if errorlevel 1 (
    echo ❌ 中文处理库安装失败
    pause
    exit /b 1
)

:: 安装其他依赖
echo [8/8] 安装其他依赖...
python -m pip install tqdm psutil numpy
python -m pip install huggingface-hub datasets
if errorlevel 1 (
    echo ❌ 其他依赖安装失败
    pause
    exit /b 1
)

:: 可选：TensorRT (通常需要NVIDIA开发者账号)
echo.
echo [可选] 安装TensorRT加速 (需要NVIDIA开发者账号)
set /p install_tensorrt="是否安装TensorRT? (y/N): "
if /i "%install_tensorrt%"=="y" (
    echo 安装TensorRT...
    python -m pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com
    python -m pip install pycuda
    if errorlevel 1 (
        echo ⚠️  TensorRT安装失败，但不影响基本功能
    ) else (
        echo ✅ TensorRT安装成功
    )
)

echo.
echo ===============================================================
echo 🎉 安装完成！
echo ===============================================================
echo.
echo 接下来请运行测试脚本验证安装:
echo   python test_system.py
echo.
echo 然后开始使用:
echo   python main.py 视频文件.mp4
echo.
echo 或使用批量处理:
echo   python main.py --input-dir ./videos --output-dir ./subtitles
echo.
echo ===============================================================
pause
