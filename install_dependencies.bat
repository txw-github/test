@echo off
chcp 65001 > nul
title 中文电视剧字幕工具 - 依赖安装

echo ========================================
echo 中文电视剧字幕工具依赖安装 - RTX 3060 Ti版
echo ========================================
echo.

echo [INFO] 开始安装依赖包...
echo.

echo [1/10] 更新pip...
python -m pip install --upgrade pip

echo.
echo [2/10] 安装PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/10] 安装Whisper相关...
pip install openai-whisper
pip install faster-whisper

echo.
echo [4/10] 安装音频处理库...
pip install moviepy
pip install soundfile
pip install librosa
pip install pyloudnorm

echo.
echo [5/10] 安装文本处理库...
pip install jieba
pip install zhon

echo.
echo [6/10] 安装进度条和工具...
pip install tqdm
pip install psutil

echo.
echo [7/10] 安装可选模型库...
echo 正在安装FunASR...
pip install funasr[all] -i https://pypi.org/simple/

echo.
echo [8/10] 尝试安装TensorRT (可选)...
pip install tensorrt
pip install pycuda

echo.
echo [9/10] 安装其他依赖...
pip install transformers
pip install datasets
pip install numpy
pip install scipy

echo.
echo [10/10] 安装完成检查...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

echo.
echo ========================================
echo 依赖安装完成！
echo.
echo 接下来请运行: python test_system.py
echo ========================================
pause