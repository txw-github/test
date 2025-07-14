
@echo off
chcp 65001 > nul
title RTX 3060 Ti 视频转字幕工具 - 环境安装

echo ========================================
echo RTX 3060 Ti 视频转字幕工具环境安装
echo ========================================
echo.

echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8-3.11
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo [✓] Python环境正常
echo.

echo 检查显卡驱动...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到NVIDIA驱动，将使用CPU模式
) else (
    echo [✓] NVIDIA驱动正常
)
echo.

echo [1/6] 升级pip...
python -m pip install --upgrade pip
echo.

echo [2/6] 安装PyTorch (CUDA 12.1)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

echo [3/6] 安装Whisper模型...
pip install faster-whisper
pip install openai-whisper
echo.

echo [4/6] 安装音视频处理...
pip install moviepy
echo.

echo [5/6] 安装中文处理...
pip install jieba
echo.

echo [6/6] 安装其他工具...
pip install tqdm psutil
echo.

echo 测试安装结果...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo.

if exist test_system.py (
    python test_system.py
) else (
    echo 基本依赖安装完成！
)

echo.
echo ========================================
echo 安装完成！
echo 使用方法: python main.py 视频.mp4
echo ========================================
pause
