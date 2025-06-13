
#!/usr/bin/env python3
"""
系统测试脚本 - 检查环境是否正确配置
"""

import sys
import os
import subprocess
import traceback

def test_python():
    """测试Python环境"""
    print("=" * 50)
    print("测试Python环境")
    print("=" * 50)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    return True

def test_pytorch():
    """测试PyTorch"""
    print("\n" + "=" * 50)
    print("测试PyTorch")
    print("=" * 50)
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    except Exception as e:
        print(f"PyTorch测试失败: {e}")
        return False

def test_whisper():
    """测试Whisper"""
    print("\n" + "=" * 50)
    print("测试Whisper")
    print("=" * 50)
    try:
        import whisper
        print(f"Whisper版本: {whisper.__version__}")
        print("Whisper可用")
        return True
    except Exception as e:
        print(f"Whisper不可用: {e}")
        return False

def test_faster_whisper():
    """测试Faster-Whisper"""
    print("\n" + "=" * 50)
    print("测试Faster-Whisper")
    print("=" * 50)
    try:
        from faster_whisper import WhisperModel
        print("Faster-Whisper可用")
        return True
    except Exception as e:
        print(f"Faster-Whisper不可用: {e}")
        return False

def test_transformers():
    """测试Transformers"""
    print("\n" + "=" * 50)
    print("测试Transformers")
    print("=" * 50)
    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
        return True
    except Exception as e:
        print(f"Transformers不可用: {e}")
        return False

def test_modelscope():
    """测试ModelScope"""
    print("\n" + "=" * 50)
    print("测试ModelScope (FireRedASR)")
    print("=" * 50)
    try:
        import modelscope
        print(f"ModelScope版本: {modelscope.__version__}")
        return True
    except Exception as e:
        print(f"ModelScope不可用: {e}")
        return False

def test_tensorrt():
    """测试TensorRT"""
    print("\n" + "=" * 50)
    print("测试TensorRT")
    print("=" * 50)
    try:
        import tensorrt as trt
        print(f"TensorRT版本: {trt.__version__}")
        return True
    except Exception as e:
        print(f"TensorRT不可用: {e}")
        return False

def test_moviepy():
    """测试MoviePy"""
    print("\n" + "=" * 50)
    print("测试MoviePy")
    print("=" * 50)
    try:
        import moviepy
        print(f"MoviePy版本: {moviepy.__version__}")
        return True
    except Exception as e:
        print(f"MoviePy不可用: {e}")
        return False

def test_ffmpeg():
    """测试FFmpeg"""
    print("\n" + "=" * 50)
    print("测试FFmpeg")
    print("=" * 50)
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"FFmpeg: {version_line}")
            return True
        else:
            print("FFmpeg不可用")
            return False
    except Exception as e:
        print(f"FFmpeg测试失败: {e}")
        return False

def test_audio_libs():
    """测试音频处理库"""
    print("\n" + "=" * 50)
    print("测试音频处理库")
    print("=" * 50)
    
    libs = ['librosa', 'soundfile', 'numpy', 'scipy']
    results = []
    
    for lib in libs:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'unknown')
            print(f"{lib}: {version}")
            results.append(True)
        except Exception as e:
            print(f"{lib}: 不可用 ({e})")
            results.append(False)
    
    return all(results)

def main():
    """主测试函数"""
    print("视频转字幕工具 - 系统环境测试")
    print("=" * 50)
    
    tests = [
        ("Python环境", test_python),
        ("PyTorch", test_pytorch),
        ("Whisper", test_whisper),
        ("Faster-Whisper", test_faster_whisper),
        ("Transformers", test_transformers),
        ("ModelScope", test_modelscope),
        ("TensorRT", test_tensorrt),
        ("MoviePy", test_moviepy),
        ("FFmpeg", test_ffmpeg),
        ("音频处理库", test_audio_libs),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"{name} 测试异常: {e}")
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:<20} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统环境配置正确。")
    elif passed >= total * 0.7:
        print("⚠️  大部分测试通过，系统基本可用。")
    else:
        print("❌ 多项测试失败，请检查安装。")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
