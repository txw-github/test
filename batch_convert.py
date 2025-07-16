
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量视频转字幕工具
支持处理整个目录下的所有视频文件
"""

import os
import sys

def get_user_input():
    """获取用户输入"""
    print("\n" + "="*50)
    print("     批量视频转字幕工具")
    print("="*50)
    
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
    output_dir = input("请输入字幕输出文件夹路径: ").strip().strip('"')
    if not output_dir:
        output_dir = os.path.join(input_dir, "subtitles")
        print(f"使用默认输出目录: {output_dir}")
    
    # 选择模型
    print("\n可选择的模型:")
    models = {
        "1": ("tiny", "最快，质量较低，显存占用最小"),
        "2": ("base", "推荐，平衡性能和质量"),
        "3": ("small", "较快，质量好"),
        "4": ("faster-base", "推荐，比base快5倍"),
        "5": ("faster-small", "快速，质量好"),
        "6": ("chinese-whisper-small", "中文优化小模型"),
        "7": ("chinese-whisper-base", "中文优化基础模型")
    }
    
    for key, (model, desc) in models.items():
        print(f"{key}. {model} - {desc}")
    
    while True:
        choice = input("\n请选择模型 (1-7, 默认2): ").strip()
        if not choice:
            choice = "2"
        if choice in models:
            model = models[choice][0]
            break
        print("❌ 请输入有效的选项 (1-7)")
    
    # 选择设备
    print("\n选择计算设备:")
    print("1. auto - 自动检测 (推荐)")
    print("2. cuda - 强制使用GPU")
    print("3. cpu - 使用CPU")
    
    device_choice = input("请选择设备 (1-3, 默认1): ").strip()
    devices = {"1": "auto", "2": "cuda", "3": "cpu"}
    device = devices.get(device_choice, "auto")
    
    return input_dir, output_dir, model, device

def main():
    """主函数"""
    try:
        input_dir, output_dir, model, device = get_user_input()
        
        print(f"\n" + "="*50)
        print("开始批量转换...")
        print(f"📂 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        print(f"🤖 使用模型: {model}")
        print(f"🔧 使用设备: {device}")
        print("="*50)
        
        # 构建命令
        command = [
            sys.executable, "main.py",
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--model", model,
            "--device", device
        ]
        
        # 执行命令
        import subprocess
        result = subprocess.run(command, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print("\n✅ 批量转换完成！")
        else:
            print("\n❌ 批量转换失败！")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户取消操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
