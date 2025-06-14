
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装测试脚本
检查所有依赖是否正确安装
"""

import sys
import traceback

def test_pytorch():
    """测试PyTorch和CUDA"""
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA可用，版本: {torch.version.cuda}")
            print(f"✓ GPU设备: {torch.cuda.get_device_name(0)}")
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU显存: {memory:.1f}GB")
            return True
        else:
            print("✗ CUDA不可用")
            return False
    except Exception as e:
        print(f"✗ PyTorch测试失败: {e}")
        return False

def test_whisper():
    """测试Whisper"""
    try:
        import whisper
        print("✓ OpenAI Whisper可用")
        
        from faster_whisper import WhisperModel
        print("✓ Faster-Whisper可用")
        return True
    except Exception as e:
        print(f"✗ Whisper测试失败: {e}")
        return False

def test_audio_processing():
    """测试音频处理库"""
    try:
        import librosa
        import soundfile
        import scipy
        import numpy
        print("✓ 音频处理库可用")
        return True
    except Exception as e:
        print(f"✗ 音频处理库测试失败: {e}")
        return False

def test_video_processing():
    """测试视频处理库"""
    try:
        from moviepy.editor import VideoFileClip
        print("✓ 视频处理库可用")
        return True
    except Exception as e:
        print(f"✗ 视频处理库测试失败: {e}")
        return False

def test_fireredasr():
    """测试FireRedASR"""
    try:
        from modelscope.models.audio.asr import FireRedAsr
        print("✓ FireRedASR可用")
        return True
    except Exception as e:
        print(f"✗ FireRedASR测试失败: {e}")
        return False

def test_tensorrt():
    """测试TensorRT"""
    try:
        import tensorrt
        import pycuda.driver
        print("✓ TensorRT加速可用")
        return True
    except Exception as e:
        print(f"✗ TensorRT不可用: {e}")
        return False

def main():
    print("开始测试安装环境...")
    print("=" * 50)
    
    results = []
    results.append(test_pytorch())
    results.append(test_whisper())
    results.append(test_audio_processing())
    results.append(test_video_processing())
    results.append(test_fireredasr())
    results.append(test_tensorrt())
    
    print("=" * 50)
    
    success_count = sum(results)
    total_count = len(results)
    
    if success_count >= 4:  # 前4个是必需的
        print(f"✓ 环境测试通过！({success_count}/{total_count})")
        print("您可以开始使用音频转文字功能了")
        return True
    else:
        print(f"✗ 环境测试失败！({success_count}/{total_count})")
        print("请检查安装步骤或重新运行setup_environment.bat")
        return False

if __name__ == "__main__":
    main()
    input("\n按任意键退出...")
