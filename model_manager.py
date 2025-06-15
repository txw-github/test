
#!/usr/bin/env python3
"""
统一模型管理器 - 支持多种ASR模型
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    model_type: str  # whisper, funasr, fireredasr, sensevoice
    size: str  # tiny, small, base, large
    language: str
    precision: str
    tensorrt_support: bool
    recommended_gpu: List[str]
    min_vram_gb: float
    download_url: Optional[str] = None
    local_path: Optional[str] = None

class ModelRegistry:
    """模型注册表"""
    
    def __init__(self):
        self.models = self._init_models()
    
    def _init_models(self) -> Dict[str, ModelInfo]:
        """初始化支持的模型列表"""
        models = {}
        
        # Whisper系列
        whisper_models = [
            ("tiny", 0.5, ["GTX 1060", "RTX 2060"]),
            ("base", 1.0, ["GTX 1660", "RTX 3060"]),
            ("small", 2.0, ["RTX 3060", "RTX 3060 Ti"]),
            ("medium", 5.0, ["RTX 3070", "RTX 4060"]),
            ("large", 10.0, ["RTX 3080", "RTX 4070"])
        ]
        
        for size, vram, gpus in whisper_models:
            models[size] = ModelInfo(
                name=size,
                model_type="whisper",
                size=size,
                language="multilingual",
                precision="fp16",
                tensorrt_support=True,
                recommended_gpu=gpus,
                min_vram_gb=vram
            )
            
            # Faster-Whisper版本
            models[f"faster-{size}"] = ModelInfo(
                name=f"faster-{size}",
                model_type="faster-whisper",
                size=size,
                language="multilingual",
                precision="fp16",
                tensorrt_support=True,
                recommended_gpu=gpus,
                min_vram_gb=vram * 0.7  # Faster-Whisper更省内存
            )
        
        # FunASR系列
        funasr_models = [
            ("funasr-paraformer", "paraformer", 4.0, ["RTX 3060 Ti", "RTX 3070"]),
            ("funasr-conformer", "conformer", 6.0, ["RTX 3070", "RTX 4060"])
        ]
        
        for name, arch, vram, gpus in funasr_models:
            models[name] = ModelInfo(
                name=name,
                model_type="funasr",
                size="base",
                language="chinese",
                precision="fp16",
                tensorrt_support=True,
                recommended_gpu=gpus,
                min_vram_gb=vram
            )
        
        # FireRedASR系列
        firered_models = [
            ("fireredasr-small", "small", 2.0, ["RTX 3060", "RTX 3060 Ti"]),
            ("fireredasr-base", "base", 4.0, ["RTX 3060 Ti", "RTX 3070"]),
            ("fireredasr-large", "large", 8.0, ["RTX 3080", "RTX 4070"])
        ]
        
        for name, size, vram, gpus in firered_models:
            models[name] = ModelInfo(
                name=name,
                model_type="fireredasr",
                size=size,
                language="chinese",
                precision="fp16",
                tensorrt_support=True,
                recommended_gpu=gpus,
                min_vram_gb=vram
            )
        
        # SenseVoice系列
        sensevoice_models = [
            ("sensevoice-small", "small", 3.0, ["RTX 3060 Ti", "RTX 3070"]),
            ("sensevoice-large", "large", 6.0, ["RTX 3070", "RTX 4060"])
        ]
        
        for name, size, vram, gpus in sensevoice_models:
            models[name] = ModelInfo(
                name=name,
                model_type="sensevoice",
                size=size,
                language="chinese",
                precision="fp16",
                tensorrt_support=True,
                recommended_gpu=gpus,
                min_vram_gb=vram
            )
        
        return models
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.models.get(model_name)
    
    def get_recommended_models(self, gpu_name: str = None, vram_gb: float = None) -> List[ModelInfo]:
        """获取推荐模型"""
        if not gpu_name and not vram_gb:
            # 自动检测GPU
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                pass
        
        recommended = []
        for model in self.models.values():
            # 检查GPU兼容性
            gpu_compatible = True
            if gpu_name:
                gpu_compatible = any(gpu in gpu_name for gpu in model.recommended_gpu)
            
            # 检查显存要求
            vram_compatible = True
            if vram_gb:
                vram_compatible = model.min_vram_gb <= vram_gb
            
            if gpu_compatible and vram_compatible:
                recommended.append(model)
        
        # 按显存要求排序
        return sorted(recommended, key=lambda x: x.min_vram_gb)
    
    def get_models_by_type(self, model_type: str) -> List[ModelInfo]:
        """按类型获取模型"""
        return [model for model in self.models.values() if model.model_type == model_type]

class ModelManager:
    """统一模型管理器"""
    
    def __init__(self, config_path: str = "model_config.json"):
        self.config_path = config_path
        self.registry = ModelRegistry()
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """加载配置"""
        default_config = {
            "default_model": "faster-base",
            "auto_optimize": True,
            "tensorrt_enabled": True,
            "precision": "fp16",
            "cache_models": True,
            "models_path": "./models",
            "preferred_models": {
                "RTX 3060 Ti": ["faster-base", "funasr-paraformer", "fireredasr-base"],
                "RTX 3060": ["faster-base", "small", "fireredasr-small"],
                "RTX 3070": ["faster-large", "funasr-paraformer", "fireredasr-base"],
                "RTX 4060": ["medium", "funasr-conformer", "fireredasr-large"]
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}")
        
        return default_config
    
    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
    
    def get_optimal_model(self, task: str = "transcription", language: str = "zh") -> str:
        """获取最佳模型"""
        try:
            import torch
            if not torch.cuda.is_available():
                return "tiny"  # CPU模式使用最小模型
            
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"🎯 检测到GPU: {gpu_name} ({vram_gb:.1f}GB)")
            
            # 根据GPU选择首选模型
            for gpu_key, models in self.config["preferred_models"].items():
                if gpu_key in gpu_name:
                    for model in models:
                        model_info = self.registry.get_model_info(model)
                        if model_info and model_info.min_vram_gb <= vram_gb * 0.8:  # 留20%余量
                            logger.info(f"✅ 推荐模型: {model}")
                            return model
            
            # 降级选择
            recommended = self.registry.get_recommended_models(gpu_name, vram_gb * 0.8)
            if recommended:
                best_model = recommended[0].name
                logger.info(f"💡 自动选择模型: {best_model}")
                return best_model
            
            return "tiny"  # 兜底选择
            
        except Exception as e:
            logger.warning(f"模型选择失败: {e}")
            return self.config["default_model"]
    
    def list_available_models(self) -> List[Dict]:
        """列出可用模型"""
        models = []
        for name, info in self.registry.models.items():
            models.append({
                "name": name,
                "type": info.model_type,
                "size": info.size,
                "language": info.language,
                "vram_gb": info.min_vram_gb,
                "tensorrt": info.tensorrt_support,
                "recommended_gpu": info.recommended_gpu
            })
        
        return sorted(models, key=lambda x: (x["type"], x["vram_gb"]))
    
    def validate_model(self, model_name: str) -> bool:
        """验证模型"""
        model_info = self.registry.get_model_info(model_name)
        if not model_info:
            return False
        
        # 检查系统兼容性
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if model_info.min_vram_gb > vram_gb:
                    logger.warning(f"⚠️ 模型 {model_name} 需要 {model_info.min_vram_gb}GB 显存，当前只有 {vram_gb:.1f}GB")
                    return False
        except:
            pass
        
        return True
    
    def install_model(self, model_name: str) -> bool:
        """安装模型"""
        model_info = self.registry.get_model_info(model_name)
        if not model_info:
            logger.error(f"未知模型: {model_name}")
            return False
        
        if not self.validate_model(model_name):
            return False
        
        logger.info(f"🔄 准备安装模型: {model_name}")
        # 这里可以添加实际的模型下载逻辑
        return True

# 全局模型管理器实例
model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    """获取模型管理器"""
    return model_manager
