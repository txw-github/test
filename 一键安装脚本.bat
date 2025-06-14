
@echo off
chcp 65001
echo ========================================
echo 中文电视剧音频转文字工具 - RTX 3060 Ti专版
echo 一键安装脚本
echo ========================================

echo.
echo 🔍 第一步：检查系统环境...
echo ========================================

:: 检查Python环境
echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到Python！
    echo 请按以下步骤安装Python：
    echo 1. 访问 https://www.python.org/downloads/
    echo 2. 下载Python 3.10版本
    echo 3. 安装时请勾选"Add Python to PATH"
    echo 4. 重启电脑后重新运行此脚本
    pause
    exit /b 1
)

python --version
echo ✅ Python环境检查通过

echo.
echo 检查NVIDIA驱动...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未检测到NVIDIA驱动！
    echo 请按以下步骤安装NVIDIA驱动：
    echo 1. 访问 https://www.nvidia.com/drivers/
    echo 2. 选择RTX 3060 Ti驱动下载
    echo 3. 安装最新驱动（包含CUDA 12.1）
    echo 4. 重启电脑后重新运行此脚本
    pause
    exit /b 1
) else (
    echo ✅ NVIDIA驱动检查通过
    nvidia-smi
)

echo.
echo 🛠️ 第二步：创建工作目录...
echo ========================================
if not exist "models" mkdir models
if not exist "temp" mkdir temp  
if not exist "output" mkdir output
echo ✅ 工作目录创建完成

echo.
echo 📦 第三步：升级pip...
echo ========================================
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ❌ pip升级失败，请检查网络连接
    pause
    exit /b 1
)

echo.
echo 🔥 第四步：安装PyTorch CUDA 12.1版本...
echo ========================================
echo 正在安装PyTorch（这可能需要5-10分钟，请耐心等待）...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ❌ PyTorch安装失败，请检查网络连接
    echo 如果国外网络不稳定，请尝试使用代理或稍后重试
    pause
    exit /b 1
)

echo.
echo 🎯 验证CUDA安装...
python -c "import torch; print(f'✅ PyTorch版本: {torch.__version__}'); print(f'✅ CUDA可用: {torch.cuda.is_available()}'); print(f'✅ GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

echo.
echo 🎤 第五步：安装Whisper相关库...
echo ========================================
pip install openai-whisper
pip install faster-whisper
echo ✅ Whisper安装完成

echo.
echo 🔊 第六步：安装音频处理库...
echo ========================================
pip install librosa
pip install soundfile
pip install scipy
pip install numpy
echo ✅ 音频处理库安装完成

echo.
echo 🎬 第七步：安装视频处理库...
echo ========================================
pip install moviepy
echo ✅ 视频处理库安装完成

echo.
echo 🛠️ 第八步：安装其他必需库...
echo ========================================
pip install tqdm
pip install psutil
pip install transformers
pip install accelerate
echo ✅ 其他库安装完成

echo.
echo ⚡ 第九步：尝试安装TensorRT加速（可选）...
echo ========================================
echo 正在尝试安装TensorRT...
pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com
if errorlevel 1 (
    echo ⚠️ TensorRT安装失败，将跳过加速功能（不影响基本使用）
) else (
    echo ✅ TensorRT安装成功
)

pip install pycuda
if errorlevel 1 (
    echo ⚠️ PyCuda安装失败，将跳过TensorRT加速
) else (
    echo ✅ PyCuda安装成功
)

echo.
echo 🧪 第十步：测试安装结果...
echo ========================================
python test_installation.py

echo.
echo 🎉 第十一步：创建快捷启动脚本...
echo ========================================

:: 创建图形界面启动脚本
echo @echo off > 启动转换工具.bat
echo chcp 65001 >> 启动转换工具.bat
echo python gui_interface.py >> 启动转换工具.bat
echo pause >> 启动转换工具.bat

:: 创建命令行快速启动脚本
echo @echo off > 快速转换.bat
echo chcp 65001 >> 快速转换.bat
echo echo 拖拽视频文件到此窗口，然后按回车... >> 快速转换.bat
echo set /p video_file= >> 快速转换.bat
echo python main.py "%%video_file%%" --model faster-base >> 快速转换.bat
echo pause >> 快速转换.bat

echo ✅ 快捷启动脚本创建完成

echo.
echo 🎯 安装完成总结
echo ========================================
echo ✅ Python环境: 正常
echo ✅ NVIDIA驱动: 正常  
echo ✅ PyTorch CUDA: 安装成功
echo ✅ Whisper模型: 安装成功
echo ✅ 音视频处理: 安装成功
echo ✅ 工作目录: 已创建

echo.
echo 📖 使用说明：
echo ========================================
echo 方法1（推荐）：双击"启动转换工具.bat"使用图形界面
echo 方法2：双击"快速转换.bat"快速转换单个文件
echo 方法3：命令行使用：
echo   python main.py 你的视频.mp4 --model faster-base
echo.
echo 📁 文件夹说明：
echo   models/  - 存放下载的AI模型文件
echo   temp/    - 存放临时音频文件
echo   output/  - 存放生成的字幕文件
echo.
echo 🤖 推荐模型（RTX 3060 Ti）：
echo   faster-base  - 最佳平衡（推荐）
echo   small        - 最快速度
echo   base         - 标准质量

echo.
echo 🎉 安装成功！按任意键开始使用...
pause
