@echo off
chcp 65001
echo ========================================
echo         视频转字幕工具启动器
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python环境
    echo 请确保已安装Python 3.8+
    pause
    exit /b 1
)

echo [INFO] Python环境检查通过
echo.

REM 检查依赖
echo [INFO] 检查依赖库...
python -c "import torch, whisper" >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 缺少必要依赖，正在运行安装脚本...
    call install_dependencies.bat
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)

echo [INFO] 依赖检查通过
echo.

REM 检查CUDA环境
echo [INFO] 检查CUDA环境...
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
echo.

REM 检查TensorRT引擎
echo [INFO] 检查TensorRT引擎状态...
python tensorrt_manager.py --list
echo.

echo 请选择启动模式:
echo 1. 启动图形界面 (推荐)
echo 2. 管理TensorRT引擎
echo 3. 命令行模式帮助
echo.
set /p choice="请输入选择 (1-3): "

if "%choice%"=="1" (
    echo [INFO] 启动图形界面...
    python gui_interface.py
) else if "%choice%"=="2" (
    echo.
    echo TensorRT引擎管理选项:
    echo 1. 查看引擎列表
    echo 2. 重建引擎 (funasr-paraformer)  
    echo 3. 清理无效引擎
    echo 4. RTX 3060 Ti优化
    echo.
    set /p engine_choice="请输入选择 (1-4): "

    if "!engine_choice!"=="1" (
        python tensorrt_manager.py --list
    ) else if "!engine_choice!"=="2" (
        python tensorrt_manager.py --rebuild funasr-paraformer
    ) else if "!engine_choice!"=="3" (
        python tensorrt_manager.py --clean
    ) else if "!engine_choice!"=="4" (
        python tensorrt_manager.py --optimize funasr-paraformer
    )
) else if "%choice%"=="3" (
    echo.
    echo 命令行使用方法:
    echo python main.py [视频文件] --model [模型名] --device [cuda/cpu]
    echo.
    echo 示例:
    echo python main.py video.mp4 --model funasr-paraformer --device cuda
    echo python main.py video.mp4 --model faster-base --device cuda
    echo.
) else (
    echo 无效选择，启动默认图形界面...
    python gui_interface.py
)

pause