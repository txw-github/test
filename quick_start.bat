
@echo off
title 视频转字幕工具
echo ========================================
echo 视频转字幕工具 - 快速启动
echo ========================================

:menu
echo.
echo 请选择操作：
echo 1. 系统环境测试
echo 2. 转换视频（使用base模型）
echo 3. 转换视频（使用faster-base模型）
echo 4. 转换视频（使用FireRedASR）
echo 5. 自定义参数转换
echo 6. 查看使用帮助
echo 0. 退出
echo.

set /p choice=请输入选择 (0-6): 

if "%choice%"=="0" goto exit
if "%choice%"=="1" goto test
if "%choice%"=="2" goto convert_base
if "%choice%"=="3" goto convert_faster
if "%choice%"=="4" goto convert_firered
if "%choice%"=="5" goto convert_custom
if "%choice%"=="6" goto help

echo 无效选择，请重新输入
goto menu

:test
echo 开始系统测试...
python test_system.py
pause
goto menu

:convert_base
set /p video_file=请输入视频文件路径: 
if "%video_file%"=="" (
    echo 未输入视频文件路径
    pause
    goto menu
)
echo 使用base模型转换中...
python main.py "%video_file%" --model base
pause
goto menu

:convert_faster
set /p video_file=请输入视频文件路径: 
if "%video_file%"=="" (
    echo 未输入视频文件路径
    pause
    goto menu
)
echo 使用faster-base模型转换中...
python main.py "%video_file%" --model faster-base
pause
goto menu

:convert_firered
set /p video_file=请输入视频文件路径: 
if "%video_file%"=="" (
    echo 未输入视频文件路径
    pause
    goto menu
)
echo 使用FireRedASR模型转换中...
python main.py "%video_file%" --model firered-aed
pause
goto menu

:convert_custom
set /p video_file=请输入视频文件路径: 
set /p model_name=请输入模型名称 (base/faster-base/firered-aed): 
set /p output_file=请输入输出文件路径 (可选): 

if "%video_file%"=="" (
    echo 未输入视频文件路径
    pause
    goto menu
)

if "%model_name%"=="" set model_name=base
if "%output_file%"=="" (
    echo 使用默认输出文件名
    python main.py "%video_file%" --model %model_name%
) else (
    python main.py "%video_file%" --model %model_name% --output "%output_file%"
)
pause
goto menu

:help
echo.
echo ========================================
echo 使用说明
echo ========================================
echo.
echo 支持的模型：
echo - base: 基础模型，平衡速度和质量
echo - faster-base: 更快的基础模型
echo - firered-aed: 中文优化模型
echo - large: 高质量模型（需要更多显存）
echo.
echo 支持的视频格式：
echo - MP4, AVI, MOV, MKV, WMV, FLV, WEBM
echo.
echo 输出格式：
echo - SRT字幕文件
echo.
echo 推荐设置（RTX 3060 Ti 6GB）：
echo - 中文内容：firered-aed
echo - 英文内容：faster-base
echo - 追求速度：base
echo.
pause
goto menu

:exit
echo 再见！
exit
