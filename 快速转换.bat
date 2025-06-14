
@echo off
chcp 65001
title 快速视频转字幕工具

echo ========================================
echo 🎬 快速视频转字幕工具
echo ========================================
echo.
echo 📖 使用说明：
echo 1. 拖拽视频文件到此窗口
echo 2. 按回车开始转换
echo 3. 等待处理完成
echo.
echo 💡 支持格式: MP4, MKV, AVI, MOV, WMV
echo 📁 输出位置: output文件夹
echo 🤖 使用模型: faster-base (RTX 3060 Ti推荐)
echo.

:INPUT
set /p "video_file=请输入视频文件路径（拖拽文件到此处）: "

if "%video_file%"=="" (
    echo ❌ 未输入文件路径，请重新输入
    goto INPUT
)

:: 移除路径两端的引号
set "video_file=%video_file:"=%"

if not exist "%video_file%" (
    echo ❌ 文件不存在: %video_file%
    echo 请检查文件路径是否正确
    goto INPUT
)

echo.
echo 🚀 开始转换视频: %video_file%
echo 📊 使用RTX 3060 Ti优化配置...
echo.

python main.py "%video_file%" --model faster-base --device cuda

if errorlevel 1 (
    echo.
    echo ❌ 转换失败，尝试使用备用配置...
    python main.py "%video_file%" --model small --device cuda
)

echo.
echo 🎉 转换完成！
echo 📁 字幕文件保存在 output 文件夹中
echo.

:CONTINUE
echo 是否继续转换其他文件？(y/n)
set /p continue=
if /i "%continue%"=="y" goto INPUT
if /i "%continue%"=="yes" goto INPUT

echo 👋 感谢使用！
pause
