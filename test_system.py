
#!/usr/bin/env python3
"""
系统环境测试脚本
用于验证RTX 3060 Ti环境配置是否正确
"""

import sys
import traceback

def test_python_version():
    """测试Python版本"""
    print("=" * 50)
    print("测试Python环境")
    print("=" * 50)
    print(f"Python版本: {sys.version}")
    
    version_info = sys.version_info
    if version_info.major == 3 and 8 <= version_info.minor <= 11:
        print("✓ Python版本正确")
        return True
    else:
        print("✗ Python版本不正确，需要Python 3.8-3.11")
        return False

def test_torch():
    """测试PyTorch和CUDA"""
    print("\n" + "=" * 50)
    print("测试PyTorch和CUDA")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✓ CUDA可用")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  显存: {props.total_memory / 1024**3:.1f} GB")
                
                if "3060 Ti" in props.name:
                    print("✓ 检测到RTX 3060 Ti")
                    
            # 测试GPU计算
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print("✓ GPU计算测试通过")
            return True
        else:
            print("✗ CUDA不可用")
            return False
            
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    except Exception as e:
        print(f"✗ PyTorch测试失败: {e}")
        return False

def test_whisper():
    """测试Whisper"""
    print("\n" + "=" * 50)
    print("测试Whisper")
    print("=" * 50)
    
    success = True
    
    try:
        import whisper
        print("✓ OpenAI Whisper可用")
    except ImportError:
        print("✗ OpenAI Whisper未安装")
        success = False
    
    try:
        from faster_whisper import WhisperModel
        print("✓ Faster-Whisper可用")
    except ImportError:
        print("✗ Faster-Whisper未安装")
        success = False
    
    return success

def test_audio_processing():
    """测试音频处理库"""
    print("\n" + "=" * 50)
    print("测试音频处理库")
    print("=" * 50)
    
    success = True
    
    try:
        import librosa
        print("✓ Librosa可用")
    except ImportError:
        print("✗ Librosa未安装")
        success = False
    
    try:
        import soundfile
        print("✓ SoundFile可用")
    except ImportError:
        print("✗ SoundFile未安装")
        success = False
    
    try:
        import numpy
        print("✓ NumPy可用")
    except ImportError:
        print("✗ NumPy未安装")
        success = False
    
    return success

def test_video_processing():
    """测试视频处理"""
    print("\n" + "=" * 50)
    print("测试视频处理")
    print("=" * 50)
    
    try:
        import moviepy
        print("✓ MoviePy可用")
        return True
    except ImportError:
        print("⚠ MoviePy未安装，将使用FFmpeg")
        
        # 测试FFmpeg
        import subprocess
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ FFmpeg可用")
                return True
            else:
                print("✗ FFmpeg不可用")
                return False
        except FileNotFoundError:
            print("✗ FFmpeg未安装")
            return False

def test_optional_deps():
    """测试可选依赖"""
    print("\n" + "=" * 50)
    print("测试可选依赖")
    print("=" * 50)
    
    try:
        import tensorrt
        print("✓ TensorRT可用")
    except ImportError:
        print("⚠ TensorRT未安装（可选，用于加速）")
    
    try:
        import pycuda
        print("✓ PyCuda可用")
    except ImportError:
        print("⚠ PyCuda未安装（TensorRT需要）")
    
    try:
        from transformers import pipeline
        print("✓ Transformers可用")
    except ImportError:
        print("⚠ Transformers未安装（可选）")

def main():
    """主测试函数"""
    print("RTX 3060 Ti 视频转字幕工具环境测试")
    print("此测试将验证所有必需的依赖是否正确安装")
    
    results = []
    results.append(test_python_version())
    results.append(test_torch())
    results.append(test_whisper())
    results.append(test_audio_processing())
    results.append(test_video_processing())
    
    test_optional_deps()
    
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    if all(results):
        print("✓ 所有必需组件测试通过！")
        print("系统已准备就绪，可以开始使用视频转字幕工具")
        print("\n使用示例：")
        print("python main.py 你的视频.mp4 --model faster-base")
    else:
        print("✗ 部分组件测试失败")
        print("请运行 install_dependencies.bat 安装缺失的依赖")
    
    print("\n按任意键退出...")
    input()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        traceback.print_exc()
        input("按任意键退出...")
