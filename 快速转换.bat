
@echo off
chcp 65001 >nul
title 中文电视剧音频转文字工具 - RTX 3060 Ti优化版

echo.
echo ================================================
echo    中文电视剧音频转文字工具 - 增强版
echo ================================================
echo.
echo 主要功能:
echo  ✨ 智能文本纠错 (音对字不对、同音字、形近字)
echo  🎵 音频预处理增强 (提高识别质量)
echo  🎯 专业名词识别 (电视剧场景优化)
echo  📝 智能断句标点 (自动添加标点符号)
echo  🚀 多模型支持 (Whisper、FunASR等)
echo  💻 RTX 3060 Ti优化 (TensorRT加速)
echo.

:menu
echo 请选择功能:
echo  1. 转换视频文件 (推荐)
echo  2. 只提取音频
echo  3. 系统测试
echo  4. 查看使用说明
echo  5. 退出
echo.
set /p choice=请输入选项 (1-5): 

if "%choice%"=="1" goto convert_video
if "%choice%"=="2" goto extract_audio
if "%choice%"=="3" goto system_test
if "%choice%"=="4" goto show_help
if "%choice%"=="5" goto exit
goto menu

:convert_video
echo.
echo ================================================
echo           视频转字幕 - 增强版处理
echo ================================================
echo.

set /p video_path=请输入视频文件路径 (可拖拽文件): 
if "%video_path%"=="" goto menu

rem 去除引号
set video_path=%video_path:"=%

if not exist "%video_path%" (
    echo.
    echo ❌ 错误: 视频文件不存在!
    pause
    goto menu
)

echo.
echo 选择识别模型:
echo  1. faster-base (推荐 - 快速高质量)
echo  2. base (标准质量)
echo  3. small (最快速度)
echo  4. funasr-paraformer (中文优化)
echo  5. funasr-conformer (高精度中文)
echo.
set /p model_choice=请选择模型 (1-5): 

set model=faster-base
if "%model_choice%"=="1" set model=faster-base
if "%model_choice%"=="2" set model=base
if "%model_choice%"=="3" set model=small
if "%model_choice%"=="4" set model=funasr-paraformer
if "%model_choice%"=="5" set model=funasr-conformer

echo.
echo 文本处理选项:
echo  1. 启用全部增强 (推荐 - 智能纠错+断句+标点)
echo  2. 仅基础处理 (保持原始识别结果)
echo.
set /p process_choice=请选择处理方式 (1-2): 

set process_args=
if "%process_choice%"=="2" set process_args=--no-postprocess

rem 生成输出文件名
for %%f in ("%video_path%") do set "video_name=%%~nf"
set output_file=%video_name%_字幕.srt

echo.
echo ================================================
echo 开始转换...
echo ================================================
echo.
echo 📂 输入文件: %video_path%
echo 🤖 识别模型: %model%
echo 📝 输出文件: %output_file%
echo ✨ 增强处理: %process_choice%
echo.

python main.py "%video_path%" --model %model% --output "%output_file%" %process_args%

echo.
if %errorlevel%==0 (
    echo ✅ 转换完成! 字幕文件已保存为: %output_file%
    echo.
    echo 📊 处理总结:
    echo  - 音频预处理: 已增强音频质量
    echo  - 语音识别: 使用 %model% 模型
    echo  - 文本优化: 已应用多层次纠错算法
    echo  - 智能断句: 已添加标点符号和句子分割
    echo.
) else (
    echo ❌ 转换失败! 请检查错误信息
)

pause
goto menu

:extract_audio
echo.
echo ================================================
echo              仅提取音频
echo ================================================
echo.

set /p video_path=请输入视频文件路径: 
if "%video_path%"=="" goto menu

rem 去除引号
set video_path=%video_path:"=%

if not exist "%video_path%" (
    echo ❌ 错误: 视频文件不存在!
    pause
    goto menu
)

for %%f in ("%video_path%") do set "video_name=%%~nf"
set audio_file=%video_name%_音频.wav

echo.
echo 正在提取音频...
python -c "
from main import VideoSubtitleExtractor, Config
config = Config()
extractor = VideoSubtitleExtractor(config=config)
result = extractor.extract_audio('%video_path%', '%audio_file%')
if result:
    print('✅ 音频提取完成: %audio_file%')
else:
    print('❌ 音频提取失败')
"

pause
goto menu

:system_test
echo.
echo ================================================
echo              系统环境测试
echo ================================================
echo.

python test_installation.py

pause
goto menu

:show_help
echo.
echo ================================================
echo                使用说明
echo ================================================
echo.
type 使用说明.md
echo.
pause
goto menu

:exit
echo.
echo 感谢使用! 再见~
timeout /t 2 >nul
exit
