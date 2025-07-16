
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量视频转字幕工具 - 增强版
支持处理整个目录下的所有视频文件，保持目录结构
"""

import os
import sys

def get_user_input():
    """获取用户输入"""
    print("\n" + "="*60)
    print("     RTX 3060 Ti 批量视频转字幕工具")
    print("     专为中文电视剧优化")
    print("="*60)
    
    # 获取输入目录
    while True:
        input_dir = input("\n请输入视频文件夹路径: ").strip().strip('"')
        if not input_dir:
            print("❌ 请输入有效的目录路径")
            continue
        if not os.path.exists(input_dir):
            print(f"❌ 目录不存在: {input_dir}")
            continue
        if not os.path.isdir(input_dir):
            print(f"❌ 路径不是目录: {input_dir}")
            continue
        break
    
    # 获取输出目录
    output_dir = input("请输入字幕输出文件夹路径 (留空使用默认): ").strip().strip('"')
    if not output_dir:
        output_dir = os.path.join(input_dir, "subtitles")
        print(f"使用默认输出目录: {output_dir}")
    
    # 选择模型
    print(f"\n🤖 可选择的模型:")
    models = {
        "1": ("faster-base", "推荐，速度快5倍，质量好，中文友好"),
        "2": ("base", "标准选择，稳定可靠"),
        "3": ("faster-small", "最快速度，质量较好"),
        "4": ("small", "快速处理，适合长视频"),
        "5": ("faster-large-v2", "最高质量，需要TensorRT优化"),
        "6": ("large-v2", "最高质量，OpenAI官方版本"),
        "7": ("faster-large-v3", "最新版本，最佳效果")
    }
    
    for key, (model, desc) in models.items():
        print(f"{key}. {model:<18} - {desc}")
    
    while True:
        choice = input(f"\n请选择模型 (1-7, 默认1): ").strip()
        if not choice:
            choice = "1"
        if choice in models:
            model = models[choice][0]
            break
        print("❌ 请输入有效的选项 (1-7)")
    
    # 选择设备
    print(f"\n🔧 计算设备选择:")
    print("1. auto - 自动检测 (推荐)")
    print("2. cuda - 强制使用GPU")
    print("3. cpu - 使用CPU (慢但稳定)")
    
    device_choice = input("请选择设备 (1-3, 默认1): ").strip()
    devices = {"1": "auto", "2": "cuda", "3": "cpu"}
    device = devices.get(device_choice, "auto")
    
    # 选择音频质量
    print(f"\n🎵 音频质量选择:")
    print("1. balanced - 平衡质量和速度 (推荐)")
    print("2. high - 高质量处理")
    print("3. fast - 快速处理")
    
    quality_choice = input("请选择音频质量 (1-3, 默认1): ").strip()
    qualities = {"1": "balanced", "2": "high", "3": "fast"}
    audio_quality = qualities.get(quality_choice, "balanced")
    
    # 选择优化选项
    print(f"\n🚀 优化选项:")
    print("1. 中文电视剧优化 (推荐)")
    print("2. 全部优化功能")
    print("3. 基础模式")
    
    opt_choice = input("请选择优化模式 (1-3, 默认1): ").strip()
    
    optimization_args = []
    if opt_choice == "1" or not opt_choice:
        optimization_args.append("--chinese-tv-optimized")
    elif opt_choice == "2":
        optimization_args.append("--enable-all-optimizations")
    
    return input_dir, output_dir, model, device, audio_quality, optimization_args

def display_summary(input_dir, output_dir, model, device, audio_quality, optimization_args):
    """显示配置总结"""
    print(f"\n" + "="*60)
    print("📋 配置总结")
    print("="*60)
    print(f"📂 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🤖 使用模型: {model}")
    print(f"🔧 计算设备: {device}")
    print(f"🎵 音频质量: {audio_quality}")
    print(f"🚀 优化模式: {' '.join(optimization_args) if optimization_args else '基础模式'}")
    print("="*60)
    
    # 模型建议
    if "large" in model:
        print("💡 提示: 使用大模型，建议启用TensorRT优化")
    elif "faster" in model:
        print("💡 提示: 使用Faster-Whisper，速度快5倍")
    
    confirm = input("\n确认开始处理? (Y/n): ").strip().lower()
    return confirm != 'n'

def main():
    """主函数"""
    try:
        input_dir, output_dir, model, device, audio_quality, optimization_args = get_user_input()
        
        if not display_summary(input_dir, output_dir, model, device, audio_quality, optimization_args):
            print("❌ 用户取消操作")
            return
        
        print(f"\n🎬 开始批量转换...")
        
        # 构建命令
        command = [
            sys.executable, "main.py",
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--model", model,
            "--device", device,
            "--audio-quality", audio_quality
        ]
        
        # 添加优化参数
        command.extend(optimization_args)
        
        # 执行命令
        import subprocess
        result = subprocess.run(command, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print("\n🎉 批量转换完成！")
            print(f"📁 字幕文件已保存到: {output_dir}")
            print("\n💡 下次使用提示:")
            print("  • 可以直接运行此脚本进行批量处理")
            print("  • 字幕文件与原视频文件保持相同的目录结构")
            print("  • 支持多种视频格式: MP4, MKV, AVI, MOV等")
        else:
            print("\n❌ 批量转换失败！")
            print("请检查:")
            print("  • 视频文件是否存在且格式支持")
            print("  • 是否有足够的磁盘空间")
            print("  • 模型是否下载成功")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
