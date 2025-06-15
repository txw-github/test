@echo off
chcp 65001
echo ==========================================
echo    RTX 3060 Ti 视频转字幕工具 - 一键安装
echo ==========================================
echo.

:: 检查是否以管理员身份运行
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [错误] 请以管理员身份运行此脚本！
    echo 右键点击脚本，选择"以管理员身份运行"
    pause
    exit /b 1
)

echo [1/10] 检查Python环境...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [错误] Python未安装或未添加到PATH!
    echo 请访问 https://www.python.org/downloads/ 下载Python 3.8-3.11
    echo 安装时务必勾选"Add Python to PATH"
    pause
    exit /b 1
)

echo [2/10] 检查pip工具...
pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [错误] pip工具不可用!
    echo 正在尝试修复...
    python -m ensurepip --upgrade
)

echo [3/10] 升级pip到最新版本...
python -m pip install --upgrade pip

echo [4/10] 安装PyTorch (CUDA 12.1版本)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [5/10] 安装Whisper相关组件...
pip install openai-whisper
pip install faster-whisper

echo [6/10] 安装Transformers和相关库...
pip install transformers
pip install accelerate
pip install datasets

echo [7/10] 安装音视频处理库...
pip install moviepy
pip install soundfile
pip install librosa
pip install pydub
pip install ffmpeg-python

echo [8/10] 安装中文处理库...
pip install jieba
pip install zhon
pip install cn2an

echo [9/10] 安装其他必要组件...
pip install tqdm
pip install psutil
pip install numpy
pip install scipy

echo [10/10] 安装可选加速组件...
echo 正在安装ONNX Runtime (GPU版本)...
pip install onnxruntime-gpu
pip install onnx
pip install onnxsim

echo 正在安装ModelScope (FunASR支持)...
pip install modelscope
pip install funasr

echo.
echo ==========================================
echo           安装完成！正在测试...
echo ==========================================

echo 测试PyTorch CUDA支持...
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else '不可用'); print('GPU数量:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

echo.
echo 测试Whisper安装...
python -c "import whisper; print('Whisper安装成功')"

echo.
echo 测试Faster-Whisper安装...
python -c "from faster_whisper import WhisperModel; print('Faster-Whisper安装成功')"

echo.
echo 测试中文处理库...
python -c "import jieba; print('中文处理库安装成功')"

echo.
echo ==========================================
echo            安装验证完成！
echo ==========================================
echo.
echo 您现在可以：
echo 1. 运行 "python main.py 视频文件.mp4" 开始转换
echo 2. 双击 "快速转换.bat" 使用图形界面
echo 3. 查看 "使用说明.md" 获取详细说明
echo.
echo 推荐RTX 3060 Ti使用的模型：
echo - faster-base (推荐，速度快，精度好)
echo - base (标准模型)
echo - small (最快速度)
echo.
pause