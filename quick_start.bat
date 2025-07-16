
@echo off
chcp 65001
title RTX 3060 Ti 中文视频转字幕工具

echo ===============================================================
echo    RTX 3060 Ti 中文视频转字幕工具 - 快速开始
echo ===============================================================
echo.

echo 请选择操作模式:
echo.
echo 1. 单个视频文件转换
echo 2. 批量目录处理 (推荐)
echo 3. 系统环境测试
echo 4. 查看支持的模型
echo 5. 安装/更新依赖
echo.

set /p choice="请输入选择 (1-5): "

if "%choice%"=="1" goto single_file
if "%choice%"=="2" goto batch_process
if "%choice%"=="3" goto test_system
if "%choice%"=="4" goto list_models
if "%choice%"=="5" goto install_deps

echo ❌ 无效选择
pause
exit

:single_file
echo.
echo 🎬 单个视频文件转换模式
echo.
set /p video_file="请输入视频文件路径: "
if not exist "%video_file%" (
    echo ❌ 文件不存在: %video_file%
    pause
    exit
)

echo.
echo 推荐使用中文电视剧优化模式
python main.py "%video_file%" --model faster-base --chinese-tv-optimized --audio-quality balanced
goto end

:batch_process
echo.
echo 📁 批量目录处理模式 (推荐)
echo.
python batch_convert.py
goto end

:test_system
echo.
echo 🔍 系统环境测试
echo.
python test_system.py
goto end

:list_models
echo.
echo 🤖 查看支持的模型
echo.
python main.py --list-models
goto end

:install_deps
echo.
echo 📦 安装/更新依赖
echo.
call install_dependencies.bat
goto end

:end
echo.
echo ===============================================================
echo 操作完成
echo ===============================================================
pause
