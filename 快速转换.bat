
@echo off
chcp 65001
title RTX 3060 Ti 视频转字幕工具
color 0A

echo ==========================================
echo    RTX 3060 Ti 视频转字幕工具 - 快速转换
echo ==========================================
echo.

:MENU
echo 请选择操作：
echo 1. 转换视频文件
echo 2. 批量转换
echo 3. 查看系统信息
echo 4. 测试环境
echo 5. 退出
echo.
set /p choice=请输入选项 (1-5): 

if "%choice%"=="1" goto SINGLE_CONVERT
if "%choice%"=="2" goto BATCH_CONVERT
if "%choice%"=="3" goto SYSTEM_INFO
if "%choice%"=="4" goto TEST_ENV
if "%choice%"=="5" goto EXIT
echo 无效选项，请重新选择！
goto MENU

:SINGLE_CONVERT
echo.
echo ==========================================
echo              单文件转换
echo ==========================================
echo.
set /p video_file=请输入视频文件路径（或拖拽文件到此处）: 

if not exist "%video_file%" (
    echo [错误] 文件不存在！
    pause
    goto MENU
)

echo.
echo 请选择模型：
echo 1. faster-base (推荐 - 速度快，精度好)
echo 2. base (标准模型)
echo 3. small (最快速度)
echo 4. funasr-paraformer (中文优化，需要更多内存)
echo.
set /p model_choice=请选择模型 (1-4): 

if "%model_choice%"=="1" set model=faster-base
if "%model_choice%"=="2" set model=base
if "%model_choice%"=="3" set model=small
if "%model_choice%"=="4" set model=funasr-paraformer

if "%model%"=="" (
    echo 无效选择，使用默认模型 faster-base
    set model=faster-base
)

echo.
echo 开始转换，使用模型: %model%
echo 请耐心等待...
echo.

python main.py "%video_file%" --model %model% --output "%video_file%.srt"

if %errorLevel% equ 0 (
    echo.
    echo ==========================================
    echo              转换完成！
    echo ==========================================
    echo 字幕文件保存为: %video_file%.srt
) else (
    echo.
    echo ==========================================
    echo              转换失败！
    echo ==========================================
    echo 请检查错误信息或尝试其他模型
)

pause
goto MENU

:BATCH_CONVERT
echo.
echo ==========================================
echo              批量转换
echo ==========================================
echo.
set /p folder=请输入包含视频文件的文件夹路径: 

if not exist "%folder%" (
    echo [错误] 文件夹不存在！
    pause
    goto MENU
)

echo.
echo 正在扫描视频文件...
for %%f in ("%folder%\*.mp4" "%folder%\*.avi" "%folder%\*.mkv" "%folder%\*.mov") do (
    echo 发现: %%f
    python main.py "%%f" --model faster-base --output "%%f.srt"
    echo ----------------------------------------
)

echo 批量转换完成！
pause
goto MENU

:SYSTEM_INFO
echo.
echo ==========================================
echo              系统信息
echo ==========================================
echo.
python -c "
import torch
import psutil
import platform

print('操作系统:', platform.system(), platform.release())
print('Python版本:', platform.python_version())
print('CPU:', platform.processor())
print('内存总量: {:.1f} GB'.format(psutil.virtual_memory().total / 1024**3))
print('可用内存: {:.1f} GB'.format(psutil.virtual_memory().available / 1024**3))
print()
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA版本:', torch.version.cuda)
    print('GPU名称:', torch.cuda.get_device_name(0))
    print('GPU显存: {:.1f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024**3))
else:
    print('GPU: 不可用或未正确安装CUDA')
"
echo.
pause
goto MENU

:TEST_ENV
echo.
echo ==========================================
echo              环境测试
echo ==========================================
echo.
python test_installation.py
pause
goto MENU

:EXIT
echo 感谢使用！
pause
exit
