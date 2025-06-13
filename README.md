
# 视频转字幕工具 - 多模型支持版

## 系统要求

- **操作系统**: Windows 10/11
- **显卡**: NVIDIA RTX 3060 Ti (6GB显存)
- **CUDA**: 12.1版本
- **Python**: 3.8-3.11
- **显存**: 至少4GB可用显存

## 安装步骤

### 1. 安装Python
- 从官网下载Python 3.8-3.11版本
- 安装时勾选"Add Python to PATH"
- 验证安装：在命令行输入 `python --version`

### 2. 安装CUDA和驱动
- 确保NVIDIA驱动是最新版本
- CUDA 12.1应该已经包含在驱动中
- 验证：在命令行输入 `nvidia-smi`

### 3. 安装FFmpeg（可选）
- 下载FFmpeg Windows版本
- 解压到 `C:\ffmpeg`
- 将 `C:\ffmpeg\bin` 添加到环境变量PATH

### 4. 运行安装脚本
双击运行 `install_windows.bat` 文件，它会自动安装所有依赖。

### 5. 手动安装（如果脚本失败）
```bash
# 升级pip
python -m pip install --upgrade pip

# 安装PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
pip install openai-whisper faster-whisper transformers
pip install moviepy librosa soundfile numpy scipy
pip install modelscope tqdm psutil

# 可选：TensorRT加速（需要NVIDIA账号）
pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com
pip install pycuda
```

## 使用方法

### 基本用法
```bash
# 使用默认base模型
python main.py 视频文件.mp4

# 指定输出文件
python main.py 视频文件.mp4 --output 字幕.srt

# 选择模型
python main.py 视频文件.mp4 --model large
```

### 支持的模型

#### Whisper系列
- `tiny`: 最快，精度较低，显存需求最小
- `base`: 平衡选择，适合RTX 3060 Ti
- `small`: 更高精度，需要更多显存
- `medium`: 高精度，建议有8GB+显存
- `large`: 最高精度，需要大量显存

#### Faster-Whisper系列（推荐）
- `faster-base`: 比原版base快5倍
- `faster-large`: 比原版large快5倍

#### FireRedASR系列（中文优化）
- `firered-aed`: 中文语音识别优化
- `firered-llm`: 基于大语言模型的ASR

### 参数说明

```bash
python main.py [视频文件] [选项]

选项:
  --output, -o      输出字幕文件路径 (默认: output.srt)
  --model, -m       模型选择 (默认: base)
  --device, -d      设备选择 (cuda/cpu, 默认: cuda)
  --language, -l    语言设置 (默认: zh)
  --keep-temp       保留临时音频文件
```

### 使用示例

```bash
# 处理中文电视剧，使用faster-whisper
python main.py 电视剧.mkv --model faster-base --output 电视剧字幕.srt

# 使用FireRedASR处理中文内容
python main.py 中文视频.mp4 --model firered-aed --language zh

# 使用large模型获得最佳效果
python main.py 重要视频.mp4 --model large --output 高质量字幕.srt
```

## 性能优化建议

### RTX 3060 Ti 6GB显存优化
1. **推荐模型**: `faster-base`, `base`, `firered-aed`
2. **避免使用**: `large`, `medium`（显存不足）
3. **如果显存不足**: 使用 `--device cpu` 切换到CPU模式

### 加速设置
1. **启用混合精度**: 自动启用FP16加速
2. **TensorRT加速**: 安装TensorRT后自动启用
3. **批处理**: 长音频自动分块处理

## 常见问题

### 1. CUDA内存不足
```
解决方案：
- 使用更小的模型 (tiny, base)
- 切换到CPU模式: --device cpu
- 关闭其他占用显存的程序
```

### 2. 模型下载失败
```
解决方案：
- 检查网络连接
- 使用代理或镜像源
- 手动下载模型文件
```

### 3. 音频提取失败
```
解决方案：
- 安装FFmpeg
- 检查视频文件格式
- 确保视频文件完整
```

### 4. 转录质量不佳
```
解决方案：
- 使用更大的模型
- 调整音频质量
- 使用专门的中文模型 (firered-aed)
```

## 支持的文件格式

### 视频格式
- MP4, AVI, MOV, MKV, WMV, FLV, WEBM

### 音频格式
- WAV, MP3, AAC, FLAC, OGG

### 输出格式
- SRT字幕文件

## 技术特性

- ✅ 多模型支持 (Whisper, FireRedASR)
- ✅ GPU加速 (CUDA, TensorRT)
- ✅ 自动显存管理
- ✅ 长音频分块处理
- ✅ 中文优化
- ✅ 自动音频预处理
- ✅ 错误恢复机制

## 更新日志

### v1.0.0
- 支持多种ASR模型
- 优化GPU显存使用
- 添加TensorRT加速
- 完善错误处理
- 支持长音频处理

## 许可证

MIT License

## 技术支持

如有问题请查看日志文件 `video_subtitle.log` 获取详细错误信息。
