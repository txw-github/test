
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型检查和下载脚本
"""

import sys
from model_manager import ModelManager

def main():
    """主函数"""
    manager = ModelManager()
    
    print("=" * 50)
    print("模型状态检查")
    print("=" * 50)
    
    models = manager.list_models()
    
    downloaded_models = []
    available_models = []
    
    for model in models:
        if model["downloaded"]:
            downloaded_models.append(model)
        else:
            available_models.append(model)
    
    if downloaded_models:
        print("\n✅ 已下载的模型:")
        for model in downloaded_models:
            print(f"  • {model['name']} ({model['size']}) - {model['path']}")
    
    if available_models:
        print("\n❌ 可下载的模型:")
        for model in available_models:
            print(f"  • {model['name']} ({model['size']})")
    
    print("\n" + "=" * 50)
    print("使用示例:")
    print("=" * 50)
    print("# 列出所有模型")
    print("python main.py --list-models")
    print()
    print("# 下载特定模型")
    print("python main.py --download-model whisper-base")
    print("python main.py --download-model faster-whisper-base")
    print()
    print("# 使用模型转换视频")
    print("python main.py 视频.mp4 --model base")
    print()
    
    # 推荐模型
    print("推荐模型 (RTX 3060 Ti 6GB):")
    print("• faster-whisper-base (推荐) - 快速且质量好")
    print("• whisper-base - 标准质量")
    print("• whisper-small - 速度优先")


if __name__ == "__main__":
    main()
