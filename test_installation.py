#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import importlib
import platform
import subprocess

def test_python_version():
    """测试Python版本"""
    print("🔍 检查Python版本...")
    version = platform.python_version()
    print(f"   Python版本: {version}")

    version_parts = [int(x) for x in version.split('.')]
    if version_parts[0] == 3 and 8 <= version_parts[1] <= 11:
        print("   ✅ Python版本兼容")
        return True
    else:
        print("   ❌ Python版本不兼容，建议使用3.8-3.11")
        return False

def test_package_import(package_name, display_name=None):
    """测试包导入"""
    if display_name is None:
        display_name = package_name

    try:
        importlib.import_module(package_name)
        print(f"   ✅ {display_name} 已安装")
        return True
    except ImportError:
        print(f"   ❌ {display_name} 未安装")
        return False

def test_cuda_support():
    """测试CUDA支持"""
    print("🔍 检查CUDA支持...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"   ✅ CUDA可用")
            print(f"   ℹ️  CUDA版本: {torch.version.cuda}")
            print(f"   ℹ️  GPU设备: {torch.cuda.get_device_name(0)}")

            # 检查显存
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ℹ️  GPU显存: {gpu_memory:.1f} GB")

            if gpu_memory >= 5.5:  # RTX 3060 Ti有6GB
                print("   ✅ 显存充足，适合运行所有模型")
            else:
                print("   ⚠️  显存较少，建议使用小模型")

            return True
        else:
            print("   ❌ CUDA不可用")
            return False
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def test_whisper_models():
    """测试Whisper模型"""
    print("🔍 检查Whisper模型...")

    # 测试标准Whisper
    whisper_ok = test_package_import("whisper", "OpenAI Whisper")

    # 测试Faster-Whisper
    faster_whisper_ok = test_package_import("faster_whisper", "Faster-Whisper")

    return whisper_ok or faster_whisper_ok

def test_audio_video_libs():
    """测试音视频处理库"""
    print("🔍 检查音视频处理库...")

    results = []
    results.append(test_package_import("moviepy", "MoviePy"))
    results.append(test_package_import("soundfile", "SoundFile"))
    results.append(test_package_import("librosa", "Librosa"))
    results.append(test_package_import("pydub", "Pydub"))

    return any(results)

def test_chinese_processing():
    """测试中文处理库"""
    print("🔍 检查中文处理库...")

    results = []
    results.append(test_package_import("jieba", "结巴分词"))
    results.append(test_package_import("zhon", "中文字符处理"))

    return all(results)

def test_optional_acceleration():
    """测试可选加速库"""
    print("🔍 检查可选加速库...")

    # ONNX Runtime
    onnx_ok = test_package_import("onnxruntime", "ONNX Runtime")

    # FunASR
    funasr_ok = test_package_import("funasr", "FunASR")

    # ModelScope
    modelscope_ok = test_package_import("modelscope", "ModelScope")

    return True  # 这些是可选的

def test_system_resources():
    """测试系统资源"""
    print("🔍 检查系统资源...")

    try:
        import psutil

        # 内存检查
        memory = psutil.virtual_memory()
        total_gb = memory.total / 1024**3
        available_gb = memory.available / 1024**3

        print(f"   ℹ️  总内存: {total_gb:.1f} GB")
        print(f"   ℹ️  可用内存: {available_gb:.1f} GB")

        if available_gb >= 4:
            print("   ✅ 内存充足")
            memory_ok = True
        else:
            print("   ⚠️  可用内存较少，可能影响大模型运行")
            memory_ok = False

        # 磁盘空间检查
        disk = psutil.disk_usage('.')
        free_gb = disk.free / 1024**3

        print(f"   ℹ️  可用磁盘空间: {free_gb:.1f} GB")

        if free_gb >= 10:
            print("   ✅ 磁盘空间充足")
            disk_ok = True
        else:
            print("   ⚠️  磁盘空间不足，可能无法下载大模型")
            disk_ok = False

        return memory_ok and disk_ok

    except ImportError:
        print("   ❌ psutil未安装，无法检查系统资源")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("RTX 3060 Ti 视频转字幕工具 - 环境测试")
    print("=" * 50)
    print()

    test_results = []

    # 核心测试
    test_results.append(("Python版本", test_python_version()))
    test_results.append(("CUDA支持", test_cuda_support()))
    test_results.append(("Whisper模型", test_whisper_models()))
    test_results.append(("音视频处理", test_audio_video_libs()))
    test_results.append(("中文处理", test_chinese_processing()))
    test_results.append(("系统资源", test_system_resources()))

    print()

    # 可选测试
    print("🔍 检查可选功能...")
    test_optional_acceleration()

    print()
    print("=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    all_passed = True
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<15} {status}")
        if not result:
            all_passed = False

    print()

    if all_passed:
        print("🎉 所有测试通过！环境配置正常，可以开始使用。")
        print()
        print("建议使用的命令：")
        print("  python main.py 视频.mp4 --model faster-base")
        print()
        print("RTX 3060 Ti 推荐模型：")
        print("  - faster-base: 速度快，精度好 (推荐)")
        print("  - base: 标准模型")
        print("  - small: 最快速度")
    else:
        print("❌ 部分测试失败，请检查安装：")
        print("  1. 运行 '一键安装脚本.bat' 重新安装")
        print("  2. 确保以管理员身份运行")
        print("  3. 检查网络连接")
        print("  4. 确保NVIDIA驱动已安装")

    print()
    input("按任意键退出...")

if __name__ == "__main__":
    main()