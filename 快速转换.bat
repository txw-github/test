@echo off
chcp 65001 > nul
echo 快速视频转字幕工具
echo.

if "%~1"=="" (
    echo 请将视频文件拖拽到此批处理文件上
    echo.
    echo 或者选择以下选项:
    echo 1. 检查模型状态
    echo 2. 下载推荐模型
    echo 3. 退出
    echo.
    set /p choice=请选择 (1-3): 

    if "%choice%"=="1" (
        python check_models.py
        pause
        exit /b
    )

    if "%choice%"=="2" (
        echo 正在下载推荐模型...
        python main.py --download-model faster-whisper-base
        pause
        exit /b
    )

    exit /b
)

echo 正在转换: %~1
echo.

:: 检查模型是否存在，如果不存在则自动下载
python main.py "%~1" --model base

echo.
echo 转换完成! 字幕文件: %~n1.srt
pause