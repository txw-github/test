
# Windows RTX 3060 Ti 视频转字幕工具完整部署指南

## 第一步：安装Python 3.10

1. 访问 https://www.python.org/downloads/windows/
2. 下载 Python 3.10.11 (Windows installer 64-bit)
3. 运行安装程序，**重要**：勾选 "Add Python to PATH"
4. 选择 "Customize installation"
5. 勾选 "pip", "py launcher", "for all users"
6. 点击 "Install"

验证安装：
```cmd
python --version
pip --version
```

## 第二步：安装NVIDIA驱动和CUDA

1. 更新NVIDIA显卡驱动到最新版本
   - 访问 https://www.nvidia.com/drivers/
   - 下载并安装最新驱动

2. 验证CUDA 12.1已安装：
```cmd
nvidia-smi
```

## 第三步：安装FFmpeg（必需）

1. 下载FFmpeg：https://github.com/BtbN/FFmpeg-Builds/releases
2. 下载 `ffmpeg-master-latest-win64-gpl.zip`
3. 解压到 `C:\ffmpeg`
4. 添加环境变量：
   - 打开系统属性 -> 高级系统设置 -> 环境变量
   - 在系统变量PATH中添加：`C:\ffmpeg\bin`

验证安装：
```cmd
ffmpeg -version
```

## 第四步：创建项目目录

```cmd
mkdir C:\video-subtitle
cd C:\video-subtitle
```

## 第五步：安装Python依赖

创建并运行安装脚本：
