
@echo off
chcp 65001
echo ========================================
echo 中文电视剧音频转文字工具
echo ========================================

echo 可用的模型：
echo 1. tiny - 最快速度，适合测试
echo 2. base - 平衡选择，推荐日常使用
echo 3. small - 较好质量，速度较快
echo 4. faster-base - 比base快5倍（推荐）
echo 5. faster-large - 最高质量，需要更多显存
echo 6. firered-aed - 中文优化模型

echo.
set /p model_choice="请选择模型（输入数字1-6，默认2）: "

if "%model_choice%"=="" set model_choice=2
if "%model_choice%"=="1" set model_name=tiny
if "%model_choice%"=="2" set model_name=base
if "%model_choice%"=="3" set model_name=small
if "%model_choice%"=="4" set model_name=faster-base
if "%model_choice%"=="5" set model_name=faster-large
if "%model_choice%"=="6" set model_name=firered-aed

echo.
echo 当前目录的视频文件：
dir *.mp4 *.mkv *.avi *.mov /b 2>nul
echo.

set /p video_file="请输入视频文件名（包含扩展名）: "

if not exist "%video_file%" (
    echo 错误：文件 %video_file% 不存在！
    pause
    exit /b 1
)

echo.
set /p output_file="请输入输出字幕文件名（默认：字幕.srt）: "
if "%output_file%"=="" set output_file=字幕.srt

echo.
echo 开始转换...
echo 使用模型: %model_name%
echo 输入文件: %video_file%
echo 输出文件: %output_file%
echo.

python main.py "%video_file%" --model %model_name% --output "%output_file%" --language zh

echo.
if exist "%output_file%" (
    echo ✓ 转换完成！字幕文件已保存为: %output_file%
) else (
    echo ✗ 转换失败，请检查日志文件: video_subtitle.log
)

pause
