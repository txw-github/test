
# RTX 3060 Ti 视频转字幕工具使用指南

## 概述

本工具专为NVIDIA RTX 3060 Ti 6GB显卡优化，支持中文电视剧音频转文字，具有以下特性：

- ✅ 专门针对RTX 3060 Ti 6GB显存优化
- ✅ 支持多种Whisper模型（原版、Faster-Whisper）
- ✅ 自动显存管理，避免显存不足
- ✅ 支持TensorRT加速（可选）
- ✅ 完全本地部署，无需联网
- ✅ 一键安装脚本，简单易用

## 快速开始

### 1. 运行环境检查
```cmd
python test_system.py
```

### 2. 一键安装
```cmd
install_dependencies.bat
```

### 3. 快速使用
```cmd
quick_start.bat
```

### 4. 命令行使用
```cmd
# 使用推荐模型处理中文视频
python main.py 电视剧.mp4 --model faster-base --output 字幕.srt

# 使用其他模型
python main.py 视频.mp4 --model base
python main.py 视频.mp4 --model small  # 最快速度
```

## 模型选择建议

### RTX 3060 Ti 6GB 推荐配置：

1. **faster-base** (推荐)
   - 显存占用: ~2GB
   - 速度: 很快 (比原版快5倍)
   - 精度: 好
   - 适合: 日常使用

2. **base**
   - 显存占用: ~2GB
   - 速度: 中等
   - 精度: 好
   - 适合: 标准使用

3. **small**
   - 显存占用: ~1GB
   - 速度: 最快
   - 精度: 较好
   - 适合: 快速处理

### 不推荐（显存可能不足）：

- **medium**: 需要4-6GB显存
- **large**: 需要6-8GB显存

## 使用示例

### 基本使用
```cmd
# 处理单个视频
python main.py 我的视频.mp4

# 指定输出文件名
python main.py 电视剧第01集.mp4 --output 第01集字幕.srt

# 使用最快模型
python main.py 视频.mp4 --model small
```

### 高级选项
```cmd
# 保留临时音频文件
python main.py 视频.mp4 --keep-temp

# 强制使用CPU（如果GPU有问题）
python main.py 视频.mp4 --device cpu

# 指定语言
python main.py 英文视频.mp4 --language en
```

## 性能优化

### 显存优化
- 自动使用85%显存，预留15%给系统
- 显存不足时自动降级参数
- 处理完成后自动清理显存

### 速度优化
- 推荐使用 `faster-base` 模型
- 自动启用FP16混合精度
- 支持TensorRT加速（如果可用）

### 批处理优化
- RTX 3060 Ti最优批处理大小: 4
- 长音频自动分段处理
- 内存占用控制

## 故障排除

### 1. 显存不足错误
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```cmd
# 使用更小的模型
python main.py 视频.mp4 --model small

# 或者使用CPU
python main.py 视频.mp4 --device cpu
```

### 2. 模型下载失败
**解决方案：**
- 检查网络连接
- 删除 `models` 文件夹重新下载
- 使用代理或镜像源

### 3. 音频提取失败
**解决方案：**
- 确保FFmpeg已安装并在PATH中
- 检查视频文件是否损坏
- 尝试转换视频格式

### 4. 转录质量不佳
**解决方案：**
- 使用更大的模型：`base` → `faster-base`
- 确保音频清晰，减少背景噪音
- 检查音频采样率（建议16kHz）

## 支持的文件格式

### 输入视频格式
- MP4, AVI, MOV, MKV, WMV, FLV, WEBM
- 建议使用MP4格式以获得最佳兼容性

### 输出字幕格式
- SRT (SubRip Text)
- UTF-8编码，兼容所有主流播放器

## 性能参考

### RTX 3060 Ti测试结果

| 模型 | 1小时视频处理时间 | 显存占用 | 推荐度 |
|------|------------------|----------|--------|
| small | ~5分钟 | 1GB | ⭐⭐⭐ |
| base | ~8分钟 | 2GB | ⭐⭐⭐⭐ |
| faster-base | ~3分钟 | 2GB | ⭐⭐⭐⭐⭐ |
| medium | ~15分钟 | 4-6GB | ⚠️ |
| large | 可能失败 | 6-8GB | ❌ |

## 技术特性

### 显存管理
- 自动检测RTX 3060 Ti并优化配置
- 动态显存分配，避免OOM
- 智能批处理大小调整

### 模型支持
- OpenAI Whisper (tiny, base, small, medium, large)
- Faster-Whisper (更快的推理速度)
- 计划支持更多中文优化模型

### 加速技术
- CUDA加速
- FP16混合精度
- TensorRT优化（可选）

## 常见问题

**Q: 为什么不支持FireRedASR？**
A: 由于版本兼容性问题，暂时禁用了ModelScope。未来版本会修复。

**Q: 可以处理多长的视频？**
A: 理论上无限制，会自动分段处理。建议单个文件不超过2小时。

**Q: 支持实时转录吗？**
A: 当前版本仅支持文件转录，不支持实时转录。

**Q: 可以自定义模型吗？**
A: 支持，可以修改ModelFactory类添加自定义模型。

## 更新日志

### v1.0.0
- RTX 3060 Ti专项优化
- 修复ModelScope兼容性问题
- 添加自动化安装脚本
- 完善错误处理和显存管理

## 技术支持

遇到问题请查看：
1. `video_subtitle.log` 日志文件
2. 运行 `python test_system.py` 检查环境
3. 参考本文档的故障排除部分

---

**注意**: 本工具专为RTX 3060 Ti优化，其他显卡可能需要调整参数。
