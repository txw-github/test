
@echo off
chcp 65001 >nul
echo.
echo ========================================
echo     批量视频转字幕工具
echo ========================================
echo.

set /p input_dir="请输入视频文件夹路径: "
set /p output_dir="请输入字幕输出文件夹路径: "

echo.
echo 可选择的模型:
echo 1. tiny (最快，质量较低)
echo 2. base (推荐，平衡性能)
echo 3. small (较快，质量好)
echo 4. faster-base (推荐，比base快5倍)
echo 5. faster-small (快速，质量好)
echo.

set /p model_choice="请选择模型 (1-5, 默认2): "

if "%model_choice%"=="" set model_choice=2

if "%model_choice%"=="1" set model=tiny
if "%model_choice%"=="2" set model=base
if "%model_choice%"=="3" set model=small
if "%model_choice%"=="4" set model=faster-base
if "%model_choice%"=="5" set model=faster-small

echo.
echo ========================================
echo 开始批量转换...
echo 输入目录: %input_dir%
echo 输出目录: %output_dir%
echo 使用模型: %model%
echo ========================================
echo.

python main.py --input-dir "%input_dir%" --output-dir "%output_dir%" --model %model%

echo.
echo 转换完成！按任意键退出...
pause >nul
