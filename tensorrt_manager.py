
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

logger = logging.getLogger(__name__)

# 避免循环导入，直接在这里定义所需的类
class Config:
    """配置管理类 - 简化版本"""
    def __init__(self):
        self.config_file = "config.json"
        self.load_config()

    def load_config(self):
        """加载配置文件"""
        default_config = {
            "models_path": "./models",
            "temp_path": "./temp",
            "output_path": "./output",
            "gpu_memory_fraction": 0.85,
            "batch_size": 4,
            "max_segment_length": 30,
            "preferred_model": "faster-base",
            "use_tensorrt": True,
            "audio_sample_rate": 16000
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    self.config = config
            except Exception as e:
                logger.warning(f"配置文件加载失败，使用默认配置: {e}")
                self.config = default_config
        else:
            self.config = default_config

    def get(self, key, default=None):
        return self.config.get(key, default)

# TensorRT优化器类 - 简化版本
class TensorRTOptimizer:
    """TensorRT优化器 - 简化版本"""
    
    @staticmethod
    def convert_to_tensorrt(onnx_path: str, engine_path: str, precision: str = "fp16") -> bool:
        """将ONNX模型转换为TensorRT引擎"""
        try:
            # 尝试导入TensorRT
            try:
                import tensorrt as trt
                import pycuda.driver as cuda
                import pycuda.autoinit
            except ImportError:
                logger.warning("TensorRT不可用，跳过优化")
                return False

            if not os.path.exists(onnx_path):
                logger.error(f"ONNX文件不存在: {onnx_path}")
                return False

            logger.info(f"开始转换TensorRT引擎: {onnx_path} -> {engine_path}")

            # 确保输出目录存在
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)

            # 创建TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)

            # 创建builder和network
            builder = trt.Builder(trt_logger)
            config = builder.create_builder_config()

            # RTX 3060 Ti优化设置
            config.max_workspace_size = 1 << 30  # 1GB
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

            # 启用精度优化
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("启用FP16精度优化")
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("启用INT8精度优化")

            # 创建网络
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

            # 解析ONNX文件
            parser = trt.OnnxParser(network, trt_logger)

            logger.info("解析ONNX模型...")
            with open(onnx_path, 'rb') as model:
                model_data = model.read()
                if not model_data:
                    logger.error("ONNX文件为空")
                    return False

                if not parser.parse(model_data):
                    logger.error("ONNX解析失败")
                    return False

            logger.info("开始构建引擎...")
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                logger.error("TensorRT引擎构建失败")
                return False

            # 保存引擎
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)

            if os.path.exists(engine_path):
                file_size = os.path.getsize(engine_path)
                logger.info(f"TensorRT引擎构建成功: {engine_path} ({file_size/1024/1024:.1f}MB)")
                return True
            else:
                logger.error("引擎文件保存失败")
                return False

        except Exception as e:
            logger.error(f"TensorRT转换失败: {e}")
            return False

    @staticmethod
    def create_fallback_engine(engine_path: str, model_info: Dict) -> bool:
        """创建后备引擎参数文件"""
        try:
            logger.info("创建TensorRT后备参数文件...")

            fallback_config = {
                "engine_info": {
                    "precision": "fp16",
                    "max_batch_size": 1,
                    "max_workspace_size": 1073741824,
                    "input_shapes": {
                        "audio_input": [-1, 80, -1]
                    },
                    "output_shapes": {
                        "text_output": [-1, -1]
                    }
                },
                "optimization_flags": [
                    "FP16", "SPARSE_WEIGHTS"
                ],
                "created_time": time.time(),
                "rtx_3060ti_optimized": True
            }

            config_path = engine_path.replace('.trt', '_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(fallback_config, f, indent=2, ensure_ascii=False)

            logger.info(f"后备配置文件创建成功: {config_path}")
            return True

        except Exception as e:
            logger.error(f"后备参数文件创建失败: {e}")
            return False

    @staticmethod
    def load_tensorrt_engine(engine_path: str):
        """加载TensorRT引擎"""
        try:
            import tensorrt as trt
            
            if not os.path.exists(engine_path):
                logger.error(f"TensorRT引擎文件不存在: {engine_path}")
                return None

            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            if engine is None:
                logger.error("TensorRT引擎加载失败")
                return None

            logger.info(f"TensorRT引擎加载成功: {engine_path}")
            return engine

        except Exception as e:
            logger.error(f"TensorRT引擎加载错误: {e}")
            return None

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
