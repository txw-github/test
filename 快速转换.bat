@echo off
chcp 65001 >nul
title 中文电视剧音频转文字工具 - RTX 3060 Ti优化版

echo ===============================================
echo        中文电视剧音频转文字工具 v2.0
echo        多模型支持 + TensorRT加速版本
echo ===============================================
echo.

:: 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未检测到Python环境
    echo 请先安装Python 3.8以上版本
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 检查CUDA
echo 🔍 检查CUDA环境...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 警告：未检测到NVIDIA GPU或驱动
    echo 将使用CPU模式运行
    set DEVICE=cpu
) else (
    echo ✅ 检测到NVIDIA GPU
    set DEVICE=cuda
)

:: 列出支持的模型
echo.
echo 📋 支持的模型列表：
echo [Whisper系列]
echo   1. tiny      - 最快，准确度一般 (推荐CPU)
echo   2. base      - 平衡，推荐日常使用
echo   3. small     - 较好平衡
echo   4. medium    - 高准确度 (需要6GB+显存)
echo   5. large     - 最高准确度 (需要10GB+显存)
echo.
echo [Faster-Whisper系列] (推荐)
echo   6. faster-base   - 快速+准确，推荐RTX 3060 Ti
echo   7. faster-large  - 高准确度，需要较多显存
echo.
echo [FunASR系列] (中文优化)
echo   8. funasr-paraformer - 中文优化，推荐RTX 3060 Ti
echo   9. funasr-conformer  - 高精度中文识别
echo.
echo [FireRedASR系列] (高性能中文)
echo   10. fireredasr-small - 快速中文识别
echo   11. fireredasr-base  - 平衡性能，推荐
echo   12. fireredasr-large - 高精度，需要8GB+显存
echo.

:: 用户选择模型
set /p "model_choice=请选择模型 (1-12，直接回车使用推荐): "

if "%model_choice%"=="" (
    echo 🎯 自动选择最佳模型...
    python -c "from model_manager import get_model_manager; print(get_model_manager().get_optimal_model())" > temp_model.txt
    set /p selected_model=<temp_model.txt
    del temp_model.txt
) else (
    if "%model_choice%"=="1" set selected_model=tiny
    if "%model_choice%"=="2" set selected_model=base
    if "%model_choice%"=="3" set selected_model=small
    if "%model_choice%"=="4" set selected_model=medium
    if "%model_choice%"=="5" set selected_model=large
    if "%model_choice%"=="6" set selected_model=faster-base
    if "%model_choice%"=="7" set selected_model=faster-large
    if "%model_choice%"=="8" set selected_model=funasr-paraformer
    if "%model_choice%"=="9" set selected_model=funasr-conformer
    if "%model_choice%"=="10" set selected_model=fireredasr-small
    if "%model_choice%"=="11" set selected_model=fireredasr-base
    if "%model_choice%"=="12" set selected_model=fireredasr-large
)

if "%selected_model%"=="" set selected_model=faster-base

echo ✅ 已选择模型: %selected_model%
echo.

:: 输入文件选择
echo 📁 请输入视频文件路径 (支持拖拽文件到窗口):
set /p "video_file="

:: 去除引号
set video_file=%video_file:"=%

if not exist "%video_file%" (
    echo ❌ 错误：文件不存在 "%video_file%"
    pause
    exit /b 1
)

echo ✅ 视频文件: %video_file%

:: 输出文件设置
for %%f in ("%video_file%") do set "output_file=%%~dpn%%f.srt"
echo 📝 字幕文件: %output_file%

:: 高级选项
echo.
echo ⚙️ 高级选项 (直接回车使用默认设置):
set /p "use_tensorrt=启用TensorRT加速? (y/N): "
set /p "precision=精度设置 (fp16/fp32/int8): "
set /p "batch_size=批处理大小 (1-4): "

if "%precision%"=="" set precision=fp16
if "%batch_size%"=="" set batch_size=1

:: 构建命令
set cmd=python main.py "%video_file%" --model %selected_model% --device %DEVICE% --output "%output_file%" --precision %precision% --batch-size %batch_size%

if /i "%use_tensorrt%"=="y" (
    set cmd=%cmd% --tensorrt
    echo 🚀 已启用TensorRT加速
)

echo.
echo 🚀 开始转换...
echo 命令: %cmd%
echo.

:: 执行转换
%cmd%

:: 检查结果
if exist "%output_file%" (
    echo.
    echo 🎉 转换完成！
    echo 📝 字幕文件已保存至: %output_file%
    echo.
    set /p "open_file=是否打开字幕文件? (Y/n): "
    if /i not "%open_file%"=="n" (
        start "" "%output_file%"
    )
) else (
    echo.
    echo ❌ 转换失败，请检查错误信息
)

echo.
echo 按任意键退出...
pause >nul