
@echo off
chcp 65001
echo ========================================
echo 快速开始 - 中文电视剧音频转文字
echo ========================================

echo 检查环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 请先运行 setup_environment.bat 安装环境
    pause
    exit /b 1
)

echo 环境检查通过✓
echo.

echo 当前目录的视频文件：
echo ----------------------------------------
for %%f in (*.mp4 *.mkv *.avi *.mov *.wmv) do echo %%f
echo ----------------------------------------

echo.
echo 将使用默认设置：
echo - 模型: faster-base （推荐RTX 3060 Ti）
echo - 语言: 中文
echo - 输出: output.srt

echo.
set /p video_file="请输入要转换的视频文件名: "

if not exist "%video_file%" (
    echo 错误：文件不存在！
    pause
    exit /b 1
)

echo.
echo 开始转换 %video_file% ...
python main.py "%video_file%" --model faster-base --language zh --output output.srt

echo.
if exist "output.srt" (
    echo ✓ 转换完成！字幕已保存为 output.srt
    echo 您可以将此文件重命名并与视频一起使用
) else (
    echo ✗ 转换失败，请查看错误信息或运行 start_conversion.bat 尝试其他模型
)

pause
