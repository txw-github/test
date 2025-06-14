
@echo off
chcp 65001
title 中文电视剧音频转文字工具

echo 🎬 启动中文电视剧音频转文字工具...
echo 正在加载图形界面，请稍候...

python gui_interface.py

if errorlevel 1 (
    echo.
    echo ❌ 图形界面启动失败
    echo 💡 请尝试以下解决方案：
    echo 1. 确保Python已正确安装
    echo 2. 运行"一键安装脚本.bat"安装依赖
    echo 3. 使用命令行模式：python main.py 视频文件.mp4
    echo.
    pause
)
