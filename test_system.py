#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTX 3060 Ti 环境测试脚本
"""

import sys
import os

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
        else:
            print("   ⚠️  CUDA不可用，将使用CPU模式")
            return True

    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def test_whisper():
    """测试Whisper模型"""
    print("\n🔍 检查Whisper...")
    try:
        from faster_whisper import WhisperModel
        print("   ✅ Faster-Whisper可用")
        return True
    except ImportError:
        try:
            import whisper
            print("   ✅ OpenAI Whisper可用")
            return True
        except ImportError:
            print("   ❌ Whisper模型未安装")
            return False

def test_video():
    """测试视频处理"""
    print("\n🔍 检查视频处理...")
    try:
        import moviepy
        print("   ✅ MoviePy可用")
        return True
    except ImportError:
        print("   ❌ MoviePy未安装")
        return False

def test_chinese():
    """测试中文处理"""
    print("\n🔍 检查中文处理...")
    try:
        import jieba
        print("   ✅ Jieba可用")
        return True
    except ImportError:
        print("   ⚠️  Jieba未安装，中文优化功能不可用")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("RTX 3060 Ti 视频转字幕工具 - 环境测试")
    print("=" * 50)

    results = []
    results.append(test_python())
    results.append(test_torch())
    results.append(test_whisper())
    results.append(test_video())
    results.append(test_chinese())

    print("\n" + "=" * 50)
    print("测试结果")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    if passed >= 4:  # 中文处理是可选的
        print("🎉 环境配置正常！可以开始使用。")
        print("\n使用方法:")
        print("  python main.py 视频.mp4")
        print("\n推荐模型 (RTX 3060 Ti):")
        print("  --model base    (推荐)")
        print("  --model small   (更快)")
    else:
        print("❌ 环境配置有问题，请重新安装依赖")
        print("运行: install_dependencies.bat")

    print(f"\n结果: {passed}/{total} 通过")
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()