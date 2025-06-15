
#!/usr/bin/env python3
"""
TensorRT引擎管理工具
用于管理和优化TensorRT引擎文件
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional
from main import Config, TensorRTOptimizer

logger = logging.getLogger(__name__)

class TensorRTEngineManager:
    """TensorRT引擎管理器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.models_path = self.config.get('models_path', './models')
        self.engines_path = os.path.join(self.models_path, 'tensorrt_engines')
        os.makedirs(self.engines_path, exist_ok=True)
        
    def list_available_engines(self) -> List[Dict]:
        """列出可用的TensorRT引擎"""
        engines = []
        
        for file in os.listdir(self.engines_path):
            if file.endswith('.trt'):
                engine_path = os.path.join(self.engines_path, file)
                config_path = engine_path.replace('.trt', '_config.json')
                
                engine_info = {
                    "name": file.replace('.trt', ''),
                    "path": engine_path,
                    "size_mb": os.path.getsize(engine_path) / 1024 / 1024,
                    "created_time": os.path.getctime(engine_path),
                    "valid": self._validate_engine(engine_path)
                }
                
                # 加载配置信息
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            engine_info.update(config_data.get('engine_info', {}))
                    except Exception as e:
                        logger.warning(f"配置文件读取失败 {config_path}: {e}")
                
                engines.append(engine_info)
                
        return sorted(engines, key=lambda x: x['created_time'], reverse=True)
    
    def _validate_engine(self, engine_path: str) -> bool:
        """验证引擎文件"""
        try:
            if not os.path.exists(engine_path):
                return False
            
            # 基本文件检查
            if os.path.getsize(engine_path) < 1024:
                return False
                
            # 尝试加载引擎
            engine = TensorRTOptimizer.load_tensorrt_engine(engine_path)
            return engine is not None
            
        except Exception:
            return False
    
    def rebuild_engine(self, model_name: str, precision: str = "fp16") -> bool:
        """重建TensorRT引擎"""
        try:
            logger.info(f"重建TensorRT引擎: {model_name}")
            
            # 查找ONNX文件，支持多种可能的位置
            possible_onnx_paths = [
                os.path.join(self.models_path, model_name.replace("/", "_") + ".onnx"),
                os.path.join(self.models_path, model_name + ".onnx"),
                os.path.join(self.models_path, "hub", model_name.replace("/", "_") + ".onnx"),
                os.path.join(self.models_path, "hub", model_name, "model.onnx")
            ]
            
            onnx_path = None
            for path in possible_onnx_paths:
                if os.path.exists(path):
                    onnx_path = path
                    break
            
            if not onnx_path:
                logger.warning(f"未找到ONNX文件，尝试的路径: {possible_onnx_paths}")
                logger.info("将创建后备引擎配置文件...")
                engine_path = os.path.join(self.engines_path, model_name.replace("/", "_") + ".trt")
                return TensorRTOptimizer.create_fallback_engine(engine_path, {"model": model_name})
            
            engine_path = os.path.join(self.engines_path, model_name.replace("/", "_") + ".trt")
            
            # 删除旧引擎
            if os.path.exists(engine_path):
                os.remove(engine_path)
                logger.info("删除旧引擎文件")
            
            # 创建新引擎
            success = TensorRTOptimizer.convert_to_tensorrt(onnx_path, engine_path, precision)
            
            if success:
                logger.info(f"引擎重建成功: {engine_path}")
                return True
            else:
                logger.error("引擎重建失败")
                return False
                
        except Exception as e:
            logger.error(f"引擎重建异常: {e}")
            return False
    
    def clean_invalid_engines(self) -> int:
        """清理无效的引擎文件"""
        cleaned = 0
        
        for file in os.listdir(self.engines_path):
            if file.endswith('.trt'):
                engine_path = os.path.join(self.engines_path, file)
                
                if not self._validate_engine(engine_path):
                    try:
                        os.remove(engine_path)
                        # 同时删除配置文件
                        config_path = engine_path.replace('.trt', '_config.json')
                        if os.path.exists(config_path):
                            os.remove(config_path)
                        
                        logger.info(f"删除无效引擎: {file}")
                        cleaned += 1
                        
                    except Exception as e:
                        logger.warning(f"删除引擎失败 {file}: {e}")
        
        return cleaned
    
    def get_engine_info(self, model_name: str) -> Optional[Dict]:
        """获取引擎信息"""
        engines = self.list_available_engines()
        
        for engine in engines:
            if engine['name'] == model_name.replace("/", "_"):
                return engine
        
        return None
    
    def optimize_for_rtx3060ti(self, model_name: str) -> bool:
        """为RTX 3060 Ti优化引擎"""
        try:
            logger.info(f"为RTX 3060 Ti优化引擎: {model_name}")
            
            # 使用FP16精度和优化设置重建引擎
            return self.rebuild_engine(model_name, precision="fp16")
            
        except Exception as e:
            logger.error(f"RTX 3060 Ti优化失败: {e}")
            return False
    
    def auto_optimize(self, model_name: str) -> bool:
        """自动优化模型"""
        try:
            # 检查是否已有引擎
            engine_info = self.get_engine_info(model_name.replace("/", "_"))
            if engine_info and engine_info.get('valid', False):
                logger.info(f"模型 {model_name} 已有有效的TensorRT引擎")
                return True
            
            # 尝试优化
            logger.info(f"开始为模型 {model_name} 创建TensorRT引擎...")
            return self.optimize_for_rtx3060ti(model_name)
            
        except Exception as e:
            logger.warning(f"自动优化失败: {e}")
            return False

def main():
    """命令行工具主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorRT引擎管理工具")
    parser.add_argument("--list", action="store_true", help="列出所有引擎")
    parser.add_argument("--rebuild", help="重建指定模型的引擎")
    parser.add_argument("--clean", action="store_true", help="清理无效引擎")
    parser.add_argument("--optimize", help="为RTX 3060 Ti优化指定模型")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "int8"], help="精度设置")
    
    args = parser.parse_args()
    
    manager = TensorRTEngineManager()
    
    if args.list:
        engines = manager.list_available_engines()
        print("\n=== TensorRT引擎列表 ===")
        if not engines:
            print("未找到任何引擎文件")
        else:
            for engine in engines:
                status = "✅" if engine['valid'] else "❌"
                print(f"{status} {engine['name']}")
                print(f"   文件大小: {engine['size_mb']:.1f}MB")
                print(f"   创建时间: {time.ctime(engine['created_time'])}")
                print()
    
    elif args.rebuild:
        success = manager.rebuild_engine(args.rebuild, args.precision)
        if success:
            print(f"✅ 引擎重建成功: {args.rebuild}")
        else:
            print(f"❌ 引擎重建失败: {args.rebuild}")
    
    elif args.clean:
        cleaned = manager.clean_invalid_engines()
        print(f"🧹 清理完成，删除了 {cleaned} 个无效引擎")
    
    elif args.optimize:
        success = manager.optimize_for_rtx3060ti(args.optimize)
        if success:
            print(f"🚀 RTX 3060 Ti优化完成: {args.optimize}")
        else:
            print(f"❌ 优化失败: {args.optimize}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
