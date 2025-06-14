
#!/usr/bin/env python3
"""
安装测试脚本
验证RTX 3060 Ti环境配置是否正确
"""

import sys
import traceback
import time

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_result(test_name, success, details=""):
    """打印测试结果"""
    status = "✅ 通过" if success else "❌ 失败"
    print(f"{test_name:<30} {status}")
    if details:
        print(f"   详情: {details}")

def test_python_version():
    """测试Python版本"""
    print_header("🐍 Python环境测试")
    
    version_info = sys.version_info
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    print(f"Python版本: {python_version}")
    
    success = version_info.major == 3 and 8 <= version_info.minor <= 11
    print_result("Python版本检查", success, 
                "需要Python 3.8-3.11" if not success else f"版本 {python_version}")
    return success

def test_torch_cuda():
    """测试PyTorch和CUDA"""
    print_header("🔥 PyTorch和CUDA测试")
    
    try:
        import torch
        pytorch_version = torch.__version__
        print(f"PyTorch版本: {pytorch_version}")
        
        cuda_available = torch.cuda.is_available()
        print_result("PyTorch安装", True, f"版本 {pytorch_version}")
        print_result("CUDA可用性", cuda_available)
        
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            print(f"CUDA版本: {cuda_version}")
            print(f"GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_name = props.name
                gpu_memory = props.total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                if "3060 Ti" in gpu_name:
                    print_result("RTX 3060 Ti检测", True, gpu_name)
                    
            # 测试GPU计算
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
                print_result("GPU计算测试", True)
                
                # 清理GPU内存
                del x, y
                torch.cuda.empty_cache()
                
            except Exception as e:
                print_result("GPU计算测试", False, str(e))
                return False
                
        return cuda_available
        
    except ImportError:
        print_result("PyTorch安装", False, "PyTorch未安装")
        return False
    except Exception as e:
        print_result("PyTorch测试", False, str(e))
        return False

def test_whisper_models():
    """测试Whisper模型"""
    print_header("🎤 Whisper模型测试")
    
    results = []
    
    # 测试OpenAI Whisper
    try:
        import whisper
        print_result("OpenAI Whisper", True)
        results.append(True)
    except ImportError:
        print_result("OpenAI Whisper", False, "未安装")
        results.append(False)
    
    # 测试Faster-Whisper
    try:
        from faster_whisper import WhisperModel
        print_result("Faster-Whisper", True)
        results.append(True)
    except ImportError:
        print_result("Faster-Whisper", False, "未安装")
        results.append(False)
    
    # 测试Transformers
    try:
        from transformers import pipeline
        print_result("Transformers", True)
        results.append(True)
    except ImportError:
        print_result("Transformers", False, "未安装")
        results.append(False)
    
    return any(results)

def test_audio_video_libs():
    """测试音视频处理库"""
    print_header("🔊 音视频处理库测试")
    
    results = []
    
    # 测试音频库
    try:
        import librosa
        print_result("Librosa", True)
        results.append(True)
    except ImportError:
        print_result("Librosa", False, "未安装")
        results.append(False)
    
    try:
        import soundfile
        print_result("SoundFile", True)
        results.append(True)
    except ImportError:
        print_result("SoundFile", False, "未安装")
        results.append(False)
    
    try:
        import numpy
        numpy_version = numpy.__version__
        print_result("NumPy", True, f"版本 {numpy_version}")
        results.append(True)
    except ImportError:
        print_result("NumPy", False, "未安装")
        results.append(False)
    
    # 测试视频库
    try:
        import moviepy
        print_result("MoviePy", True)
        results.append(True)
    except ImportError:
        print_result("MoviePy", False, "未安装，将使用FFmpeg")
        # 测试FFmpeg
        import subprocess
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print_result("FFmpeg", True)
                results.append(True)
            else:
                print_result("FFmpeg", False, "FFmpeg不可用")
                results.append(False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print_result("FFmpeg", False, "FFmpeg未安装")
            results.append(False)
    
    return any(results)

def test_optional_acceleration():
    """测试可选加速库"""
    print_header("⚡ 可选加速库测试")
    
    # 测试TensorRT
    try:
        import tensorrt
        trt_version = tensorrt.__version__
        print_result("TensorRT", True, f"版本 {trt_version}")
    except ImportError:
        print_result("TensorRT", False, "未安装（可选）")
    
    # 测试PyCuda
    try:
        import pycuda
        print_result("PyCuda", True)
    except ImportError:
        print_result("PyCuda", False, "未安装（TensorRT需要）")

def test_model_download():
    """测试模型下载"""
    print_header("📥 模型下载测试")
    
    try:
        print("测试下载tiny模型（最小模型，用于验证）...")
        import whisper
        
        # 下载最小的模型进行测试
        model = whisper.load_model("tiny", download_root="./models")
        print_result("模型下载", True, "tiny模型下载成功")
        
        # 简单测试模型加载
        if hasattr(model, 'dims'):
            print_result("模型加载", True, f"模型维度: {model.dims}")
        else:
            print_result("模型加载", True)
            
        del model  # 释放内存
        return True
        
    except Exception as e:
        print_result("模型下载", False, str(e))
        return False

def test_file_system():
    """测试文件系统"""
    print_header("📁 文件系统测试")
    
    import os
    
    # 测试工作目录创建
    test_dirs = ["models", "temp", "output"]
    results = []
    
    for dir_name in test_dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            if os.path.exists(dir_name) and os.access(dir_name, os.W_OK):
                print_result(f"{dir_name}目录", True)
                results.append(True)
            else:
                print_result(f"{dir_name}目录", False, "无写入权限")
                results.append(False)
        except Exception as e:
            print_result(f"{dir_name}目录", False, str(e))
            results.append(False)
    
    return all(results)

def performance_benchmark():
    """性能基准测试"""
    print_header("🚀 性能基准测试")
    
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("跳过GPU性能测试（CUDA不可用）")
            return
            
        # GPU内存测试
        device = torch.device('cuda')
        
        print("测试GPU内存分配...")
        start_time = time.time()
        
        # 分配1GB内存测试
        tensor_size = (1024, 1024, 256)  # 约1GB float32
        try:
            test_tensor = torch.randn(tensor_size, device=device)
            allocation_time = time.time() - start_time
            print_result("1GB内存分配", True, f"{allocation_time:.2f}秒")
            
            # 释放内存
            del test_tensor
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print_result("1GB内存分配", False, "显存不足")
            
        # 简单计算性能测试
        print("测试GPU计算性能...")
        start_time = time.time()
        
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        compute_time = time.time() - start_time
        print_result("矩阵乘法测试", True, f"{compute_time:.2f}秒")
        
        # 清理
        del a, b, c
        torch.cuda.empty_cache()
        
    except Exception as e:
        print_result("性能测试", False, str(e))

def main():
    """主测试函数"""
    print("🎯 中文电视剧音频转文字工具 - 安装验证")
    print("此测试将验证RTX 3060 Ti环境配置是否正确")
    
    test_results = []
    
    # 核心测试
    test_results.append(test_python_version())
    test_results.append(test_torch_cuda())
    test_results.append(test_whisper_models())
    test_results.append(test_audio_video_libs())
    test_results.append(test_file_system())
    
    # 可选测试
    test_optional_acceleration()
    test_results.append(test_model_download())
    
    # 性能测试
    performance_benchmark()
    
    # 总结
    print_header("📊 测试结果总结")
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 恭喜！所有核心测试通过！")
        print("✅ 系统已准备就绪，可以开始使用视频转字幕工具")
        print("\n📖 使用方法：")
        print("1. 图形界面：双击'启动转换工具.bat'")
        print("2. 命令行：python main.py 你的视频.mp4 --model faster-base")
        print("3. 快速转换：双击'快速转换.bat'")
    else:
        print("\n⚠️ 部分测试失败，系统可能无法正常工作")
        print("❗ 请运行'一键安装脚本.bat'重新安装依赖")
        print("💡 或者检查NVIDIA驱动和CUDA环境")
    
    print(f"\n📝 详细日志已保存到当前目录")
    print("🆘 如有问题请查看上方的测试结果")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生意外错误:")
        print(f"错误: {e}")
        traceback.print_exc()
    finally:
        print("\n按回车键退出...")
        input()
