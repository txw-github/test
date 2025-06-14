
#!/usr/bin/env python3
"""
TensorRT安装和优化脚本
适用于RTX 3060 Ti 6GB显卡
"""

import os
import sys
import subprocess
import platform

def install_tensorrt():
    """安装TensorRT相关依赖"""
    print("正在安装TensorRT和相关优化库...")
    
    packages = [
        "onnx>=1.14.0",
        "onnxruntime-gpu>=1.16.0", 
        "onnxsim>=0.4.0",
        "polygraphy",
        "pycuda>=2022.1"
    ]
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败: {e}")
    
    print("\n📝 TensorRT安装说明:")
    print("1. TensorRT需要NVIDIA开发者账号下载")
    print("2. 访问: https://developer.nvidia.com/tensorrt")
    print("3. 下载适合CUDA 12.1的TensorRT版本")
    print("4. 解压后将库文件添加到系统PATH")
    print("\n💡 或者可以使用ONNX Runtime GPU版本作为替代方案")

def optimize_system():
    """系统优化设置"""
    print("应用RTX 3060 Ti优化设置...")
    
    # Windows特定优化
    if platform.system() == "Windows":
        optimizations = [
            "set CUDA_CACHE_DISABLE=0",
            "set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True",
            "set CUDA_LAUNCH_BLOCKING=0",
            "set TORCH_CUDNN_V8_API_ENABLED=1"
        ]
        
        for opt in optimizations:
            print(f"设置: {opt}")
            os.system(opt)
    
    print("✅ 系统优化完成")

if __name__ == "__main__":
    install_tensorrt()
    optimize_system()
    print("\n🎉 安装完成！重启程序以启用TensorRT加速")
