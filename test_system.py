
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTX 3060 Ti 环境测试脚本 - 增强版
"""

import sys
import os
import subprocess

def test_python():
    """测试Python版本"""
    print("🔍 检查Python版本...")
    version = sys.version_info
    print(f"   Python版本: {version.major}.{version.minor}.{version.micro}")

    if 3.8 <= version.minor <= 3.11 and version.major == 3:
        print("   ✅ Python版本兼容")
        return True
    else:
        print("   ❌ 建议使用Python 3.8-3.11")
        return False

def test_torch():
    """测试PyTorch和CUDA"""
    print("\n🔍 检查PyTorch...")
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")

        if torch.cuda.is_available():
            print("   ✅ CUDA可用")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name}")
            print(f"   显存: {gpu_memory:.1f}GB")

            if "3060 Ti" in gpu_name:
                print("   🎯 RTX 3060 Ti检测成功！")
                return True
            elif "NVIDIA" in gpu_name:
                print("   ✅ NVIDIA GPU检测成功")
                return True
            else:
                print("   ⚠️  检测到GPU但不是NVIDIA")
                return True
        else:
            print("   ⚠️  CUDA不可用，将使用CPU模式")
            return True

    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def test_whisper():
    """测试Whisper模型"""
    print("\n🔍 检查Whisper...")
    faster_available = False
    whisper_available = False
    
    try:
        from faster_whisper import WhisperModel
        print("   ✅ Faster-Whisper可用")
        faster_available = True
    except ImportError:
        print("   ❌ Faster-Whisper未安装")
    
    try:
        import whisper
        print("   ✅ OpenAI Whisper可用")
        whisper_available = True
    except ImportError:
        print("   ❌ OpenAI Whisper未安装")
    
    return faster_available or whisper_available

def test_video():
    """测试视频处理"""
    print("\n🔍 检查视频处理...")
    try:
        import moviepy
        print("   ✅ MoviePy可用")
        
        # 测试FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   ✅ FFmpeg可用")
            else:
                print("   ⚠️  FFmpeg不可用，某些功能可能受限")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   ⚠️  FFmpeg不可用，某些功能可能受限")
        
        return True
    except ImportError:
        print("   ❌ MoviePy未安装")
        return False

def test_audio():
    """测试音频处理"""
    print("\n🔍 检查音频处理...")
    try:
        import librosa
        import soundfile
        import scipy
        print("   ✅ 高级音频处理库可用")
        return True
    except ImportError:
        print("   ⚠️  高级音频处理库未完全安装")
        return False

def test_chinese():
    """测试中文处理"""
    print("\n🔍 检查中文处理...")
    try:
        import jieba
        import jieba.posseg
        print("   ✅ Jieba分词可用")
        
        # 测试基本分词
        test_text = "这是一个中文测试句子"
        words = list(jieba.cut(test_text))
        print(f"   测试分词: {' / '.join(words)}")
        
        return True
    except ImportError:
        print("   ❌ Jieba未安装，中文优化功能不可用")
        return False

def test_tensorrt():
    """测试TensorRT"""
    print("\n🔍 检查TensorRT...")
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        print(f"   ✅ TensorRT可用 (版本: {trt.__version__})")
        return True
    except ImportError:
        print("   ⚠️  TensorRT未安装 (可选功能)")
        return False

def test_model_download():
    """测试模型下载能力"""
    print("\n🔍 检查模型下载...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        print("   ✅ Hugging Face Hub可用")
        return True
    except ImportError:
        print("   ⚠️  Hugging Face Hub未安装")
        return False

def run_performance_test():
    """运行性能测试"""
    print("\n🔍 运行性能测试...")
    try:
        import torch
        if torch.cuda.is_available():
            # 简单的GPU性能测试
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            import time
            start_time = time.time()
            for _ in range(100):
                z = torch.mm(x, y)
            torch.cuda.synchronize()
            end_time = time.time()
            
            gpu_time = (end_time - start_time) * 10  # ms per operation
            print(f"   GPU性能: {gpu_time:.2f}ms/操作")
            
            if gpu_time < 50:
                print("   ✅ GPU性能优秀")
                return True
            elif gpu_time < 100:
                print("   ⚠️  GPU性能一般")
                return True
            else:
                print("   ❌ GPU性能较差")
                return False
        else:
            print("   ⚠️  无GPU，跳过性能测试")
            return True
    except Exception as e:
        print(f"   ❌ 性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("RTX 3060 Ti 视频转字幕工具 - 环境测试")
    print("=" * 50)

    results = []
    results.append(("Python环境", test_python()))
    results.append(("PyTorch&CUDA", test_torch()))
    results.append(("Whisper模型", test_whisper()))
    results.append(("视频处理", test_video()))
    results.append(("音频处理", test_audio()))
    results.append(("中文处理", test_chinese()))
    results.append(("TensorRT", test_tensorrt()))
    results.append(("模型下载", test_model_download()))
    results.append(("性能测试", run_performance_test()))

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    # 计算通过率
    essential_tests = results[:4]  # 前4个是必需的
    optional_tests = results[4:]   # 后面的是可选的
    
    essential_passed = sum(1 for _, result in essential_tests if result)
    optional_passed = sum(1 for _, result in optional_tests if result)
    
    print(f"\n必需功能: {essential_passed}/{len(essential_tests)} 通过")
    print(f"可选功能: {optional_passed}/{len(optional_tests)} 通过")

    if essential_passed >= 3:  # 至少3个核心功能通过
        print("\n🎉 环境配置基本正常！可以开始使用。")
        print("\n推荐使用方法:")
        print("  python main.py 视频.mp4")
        print("  python main.py --input-dir ./videos --output-dir ./subtitles")
        print("\n推荐模型 (RTX 3060 Ti):")
        print("  --model faster-base    (推荐)")
        print("  --model base           (标准)")
        print("  --model faster-small   (快速)")
        print("\n中文电视剧优化:")
        print("  --chinese-tv-optimized")
        print("  --enable-all-optimizations")
    else:
        print("\n❌ 环境配置有严重问题，请重新安装依赖")
        print("运行: install_dependencies.bat")

    if optional_passed < len(optional_tests) // 2:
        print(f"\n💡 提示: 安装可选组件可以获得更好的体验")
        print("   pip install librosa soundfile scipy  # 音频增强")
        print("   pip install jieba zhon              # 中文优化")
        print("   pip install tensorrt pycuda        # TensorRT加速")

    input("\n按回车键退出...")

if __name__ == "__main__":
    # 添加numpy导入用于性能测试
    try:
        import numpy as np
    except ImportError:
        pass
    
    main()
