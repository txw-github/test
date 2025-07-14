
@echo off
chcp 65001 > nul
title 中文电视剧字幕转换工具 - RTX 3060 Ti版

echo ========================================
echo 中文电视剧字幕转换工具 - RTX 3060 Ti版
echo ========================================
echo.

:MENU
echo 请选择操作：
echo 1. 转换视频文件
echo 2. 批量转换
echo 3. 高级设置
echo 4. 系统检查
echo 5. 退出
echo.
set /p choice=请输入选择 (1-5): 

if "%choice%"=="1" goto SINGLE_CONVERT
if "%choice%"=="2" goto BATCH_CONVERT
if "%choice%"=="3" goto ADVANCED_SETTINGS
if "%choice%"=="4" goto SYSTEM_CHECK
if "%choice%"=="5" goto EXIT
goto MENU

:SINGLE_CONVERT
echo.
echo ========== 单文件转换 ==========
echo.
set /p video_file=请输入视频文件路径（可直接拖拽文件）: 

if not exist "%video_file%" (
    echo [错误] 文件不存在: %video_file%
    pause
    goto MENU
)

echo.
echo 推荐模型选择：
echo 1. faster-base (推荐，速度快、质量好)
echo 2. base (标准，兼容性好)
echo 3. funasr-paraformer (中文专用，质量最佳)
echo 4. small (最快速度)
echo.
set /p model_choice=请选择模型 (1-4，默认1): 

if "%model_choice%"=="" set model_choice=1
if "%model_choice%"=="1" set selected_model=faster-base
if "%model_choice%"=="2" set selected_model=base
if "%model_choice%"=="3" set selected_model=funasr-paraformer
if "%model_choice%"=="4" set selected_model=small

echo.
set /p output_file=请输入输出字幕文件名（默认: output.srt）: 
if "%output_file%"=="" set output_file=output.srt

echo.
echo 音频预处理质量：
echo 1. 快速 (fast)
echo 2. 平衡 (balanced, 推荐)
echo 3. 高质量 (high)
echo.
set /p quality_choice=请选择质量 (1-3，默认2): 

if "%quality_choice%"=="" set quality_choice=2
if "%quality_choice%"=="1" set audio_quality=fast
if "%quality_choice%"=="2" set audio_quality=balanced
if "%quality_choice%"=="3" set audio_quality=high

echo.
echo ========== 开始转换 ==========
echo 视频文件: %video_file%
echo 使用模型: %selected_model%
echo 输出文件: %output_file%
echo 音频质量: %audio_quality%
echo.

python main.py "%video_file%" --model %selected_model% --output "%output_file%" --audio-quality %audio_quality% --enable-audio-preprocessing --analyze-text

echo.
echo ========== 转换完成 ==========
pause
goto MENU

:BATCH_CONVERT
echo.
echo ========== 批量转换 ==========
echo 将转换当前目录下所有 .mp4 .avi .mkv 文件
echo.
set /p confirm=确认开始批量转换？ (y/n): 

if /i not "%confirm%"=="y" goto MENU

for %%f in (*.mp4 *.avi *.mkv) do (
    echo 正在转换: %%f
    python main.py "%%f" --model faster-base --audio-quality balanced --enable-audio-preprocessing
    echo %%f 转换完成
    echo.
)

echo 批量转换完成！
pause
goto MENU

:ADVANCED_SETTINGS
echo.
echo ========== 高级设置 ==========
echo.
set /p video_file=请输入视频文件路径: 

echo.
echo 高级选项：
echo 1. 启用模型融合 (需要更多显存)
echo 2. 启用音频分析
echo 3. 精度设置 fp32 (更精确但更慢)
echo 4. 保留临时文件
echo.
set /p enable_ensemble=启用模型融合？ (y/n): 
set /p enable_audio_analysis=启用音频分析？ (y/n): 
set /p use_fp32=使用fp32精度？ (y/n): 
set /p keep_temp=保留临时文件？ (y/n): 

set advanced_args=
if /i "%enable_ensemble%"=="y" set advanced_args=%advanced_args% --enable-ensemble
if /i "%enable_audio_analysis%"=="y" set advanced_args=%advanced_args% --analyze-audio
if /i "%use_fp32%"=="y" set advanced_args=%advanced_args% --precision fp32
if /i "%keep_temp%"=="y" set advanced_args=%advanced_args% --keep-temp

echo.
echo 执行命令: python main.py "%video_file%" %advanced_args%
python main.py "%video_file%" %advanced_args%

pause
goto MENU

:SYSTEM_CHECK
echo.
echo ========== 系统检查 ==========
echo.
python test_system.py
echo.
pause
goto MENU

:EXIT
echo 感谢使用！
exit /b 0
