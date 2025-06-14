@echo off
echo ========================================
echo 视频转字幕工具 - 快速使用
echo ========================================

echo 请将要转换的视频文件拖拽到此窗口，然后按回车
echo 支持格式：MP4, AVI, MOV, MKV, WMV等
echo.

set /p video_file="请输入视频文件路径: "

if "%video_file%"=="" (
    echo 未输入视频文件路径！
    pause
    exit /b 1
)

echo.
echo 选择模型：
echo 1. faster-base（推荐，速度快，精度好）
echo 2. base（标准模型）
echo 3. large（高精度，需要更多显存）
echo 4. small（速度最快，精度较低）
echo.

set /p model_choice="请选择模型 (1-4，默认1): "

if "%model_choice%"=="" set model_choice=1

if "%model_choice%"=="1" set model_name=faster-base
if "%model_choice%"=="2" set model_name=base
if "%model_choice%"=="3" set model_name=large
if "%model_choice%"=="4" set model_name=small

if not defined model_name (
    echo 无效选择，使用默认模型 faster-base
    set model_name=faster-base
)

echo.
echo 开始转换，使用模型: %model_name%
echo 视频文件: %video_file%
echo.

python main.py "%video_file%" --model %model_name% --output "%~n1.srt"

echo.
echo 转换完成！字幕文件已保存。
echo.
pause