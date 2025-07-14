
#!/usr/bin/env python3
"""
系统环境测试脚本
检查RTX 3060 Ti环境是否正确配置
"""

import sys
import os
import subprocess
import platform

def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python版本...")
    version = sys.version_info
    print(f"   Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("   ❌ 需要Python 3.8+")
        return False
    else:
        print("   ✅ Python版本符合要求")
        return True

def check_cuda():
    """检查CUDA环境"""
    print("\n🔍 检查CUDA环境...")
    
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("   ✅ CUDA可用")
            gpu_count = torch.cuda.device_count()
            print(f"   GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                
                if "3060 Ti" in gpu_name:
                    print("   🎯 检测到RTX 3060 Ti，完美匹配！")
                    return True
            return True
        else:
            print("   ❌ CUDA不可用")
            return False
            
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def check_dependencies():
    """检查关键依赖"""
    print("\n🔍 检查关键依赖...")
    
    dependencies = {
        "whisper": "OpenAI Whisper",
        "faster_whisper": "Faster Whisper",
        "moviepy": "MoviePy",
        "soundfile": "SoundFile",
        "jieba": "Jieba分词",
        "tqdm": "进度条",
        "numpy": "NumPy",
    }
    
    optional_dependencies = {
        "funasr": "FunASR (中文优化)",
        "tensorrt": "TensorRT (加速)",
        "transformers": "Transformers",
        "librosa": "Librosa",
    }
    
    all_good = True
    
    # 检查必需依赖
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - 未安装")
            all_good = False
    
    # 检查可选依赖
    print("\n   可选依赖:")
    for module, name in optional_dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ⚠️ {name} - 未安装 (可选)")
    
    return all_good

def check_ffmpeg():
    """检查FFmpeg"""
    print("\n🔍 检查FFmpeg...")
    
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ FFmpeg可用")
            return True
        else:
            print("   ❌ FFmpeg不可用")
            return False
    except FileNotFoundError:
        print("   ❌ FFmpeg未安装或不在PATH中")
        print("   💡 请确保FFmpeg已安装并添加到系统PATH")
        return False

def check_memory():
    """检查系统内存"""
    print("\n🔍 检查系统内存...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / 1024**3
        available_gb = memory.available / 1024**3
        
        print(f"   总内存: {total_gb:.1f}GB")
        print(f"   可用内存: {available_gb:.1f}GB")
        
        if total_gb >= 16:
            print("   ✅ 内存充足")
            return True
        elif total_gb >= 8:
            print("   ⚠️ 内存偏少，建议关闭其他程序")
            return True
        else:
            print("   ❌ 内存不足，可能影响性能")
            return False
    except ImportError:
        print("   ⚠️ 无法检查内存状态")
        return True

def test_simple_conversion():
    """测试简单转换功能"""
    print("\n🔍 测试基本功能...")
    
    try:
        # 创建测试音频（1秒静音）
        import numpy as np
        import soundfile as sf
        
        test_audio = np.zeros(16000, dtype=np.float32)
        test_path = "test_audio.wav"
        sf.write(test_path, test_audio, 16000)
        
        # 测试加载模型
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cpu")  # 使用CPU避免显存问题
        
        # 测试转录
        segments, info = model.transcribe(test_path)
        segments = list(segments)
        
        # 清理
        os.remove(test_path)
        
        print("   ✅ 基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"   ❌ 基本功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("========================================")
    print("中文电视剧字幕工具 - 系统环境测试")
    print("========================================")
    
    results = []
    
    results.append(check_python_version())
    results.append(check_cuda())
    results.append(check_dependencies())
    results.append(check_ffmpeg())
    results.append(check_memory())
    results.append(test_simple_conversion())
    
    print("\n========================================")
    print("测试总结:")
    print("========================================")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 所有测试通过！系统已准备就绪。")
        print("\n可以开始使用:")
        print("   python main.py 你的视频.mp4")
        print("   或运行: 快速转换.bat")
    elif passed >= total - 1:
        print("⚠️ 大部分测试通过，可以正常使用。")
        print("有些可选功能可能不可用。")
    else:
        print("❌ 系统环境有问题，请检查以上错误。")
        print("\n建议:")
        print("1. 运行 install_dependencies.bat 安装依赖")
        print("2. 更新NVIDIA驱动")
        print("3. 检查Python和CUDA安装")
    
    print(f"\n测试结果: {passed}/{total} 通过")
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
