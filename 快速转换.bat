@echo off
chcp 65001 > nul
title RTX 3060 Ti 视频转字幕工具

echo ========================================
echo RTX 3060 Ti 视频转字幕工具
echo ========================================
echo.

echo 请将视频文件拖拽到此窗口，然后按回车
echo 或输入视频文件名:
set /p video_file=

if not exist "%video_file%" (
    echo 文件不存在: %video_file%
    pause
    exit /b 1
)

echo.
echo 选择模型:
echo 1. small  (快速, 1GB显存)
echo 2. base   (推荐, 2GB显存)  
echo 3. medium (高质量, 4GB显存)
echo.
set /p model_choice=请选择 (1-3, 默认2): 

if "%model_choice%"=="" set model_choice=2
if "%model_choice%"=="1" set model_name=small
if "%model_choice%"=="2" set model_name=base
if "%model_choice%"=="3" set model_name=medium

echo.
echo 开始转换: %video_file%
echo 使用模型: %model_name%
echo.

python main.py "%video_file%" --model %model_name%

if errorlevel 1 (
    echo.
    echo 转换失败！请检查:
    echo 1. 视频文件是否损坏
    echo 2. 显存是否不足 (尝试使用small模型)
    echo 3. 依赖是否正确安装
) else (
    echo.
    echo 转换完成！字幕文件已生成。
)

pause