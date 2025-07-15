
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型管理器 - 自动下载和管理模型权重
"""

import os
import sys
import json
import shutil
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # 模型配置
        self.model_configs = {
            "whisper-tiny": {
                "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f83b26f81e3ca7a33/tiny.pt",
                "filename": "tiny.pt",
                "size": "39 MB",
                "sha256": "65147644a518d12f04e32d6f83b26f81e3ca7a33"
            },
            "whisper-base": {
                "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
                "filename": "base.pt", 
                "size": "142 MB",
                "sha256": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
            },
            "whisper-small": {
                "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19a9b9c7d9a0b8e0a7d2a6/small.pt",
                "filename": "small.pt",
                "size": "461 MB", 
                "sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19a9b9c7d9a0b8e0a7d2a6"
            },
            "whisper-medium": {
                "url": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
                "filename": "medium.pt",
                "size": "1.42 GB",
                "sha256": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1"
            },
            "whisper-large": {
                "url": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3e25c9f2f7c1e79e7e9a5f67b/large.pt",
                "filename": "large.pt",
                "size": "2.87 GB",
                "sha256": "81f7c96c852ee8fc832187b0132e569d6c3065a3e25c9f2f7c1e79e7e9a5f67b"
            },
            "faster-whisper-tiny": {
                "hf_model": "guillaumekln/faster-whisper-tiny",
                "size": "39 MB"
            },
            "faster-whisper-base": {
                "hf_model": "guillaumekln/faster-whisper-base", 
                "size": "142 MB"
            },
            "faster-whisper-small": {
                "hf_model": "guillaumekln/faster-whisper-small",
                "size": "461 MB"
            },
            "faster-whisper-medium": {
                "hf_model": "guillaumekln/faster-whisper-medium",
                "size": "1.42 GB"  
            },
            "faster-whisper-large": {
                "hf_model": "guillaumekln/faster-whisper-large",
                "size": "2.87 GB"
            },
            "funasr-paraformer": {
                "modelscope_model": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                "size": "220 MB"
            },
            "funasr-paraformer-streaming": {
                "modelscope_model": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                "size": "220 MB"
            }
        }
        
        # 模型状态文件
        self.status_file = self.models_dir / "model_status.json"
        self.load_status()
    
    def load_status(self):
        """加载模型状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    self.model_status = json.load(f)
            except:
                self.model_status = {}
        else:
            self.model_status = {}
    
    def save_status(self):
        """保存模型状态"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.model_status, f, indent=2, ensure_ascii=False)
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """检查模型是否已下载"""
        if model_name not in self.model_configs:
            return False
        
        config = self.model_configs[model_name]
        
        # 检查OpenAI Whisper模型
        if "filename" in config:
            model_path = self.models_dir / config["filename"]
            return model_path.exists()
        
        # 检查Faster-Whisper模型
        elif "hf_model" in config:
            model_dir = self.models_dir / "faster-whisper" / model_name.replace("faster-whisper-", "")
            return model_dir.exists() and len(list(model_dir.glob("*"))) > 0
        
        # 检查FunASR模型
        elif "modelscope_model" in config:
            model_dir = self.models_dir / "funasr" / model_name.replace("funasr-", "")
            return model_dir.exists() and len(list(model_dir.glob("*"))) > 0
        
        return False
    
    def download_progress_hook(self, block_num, block_size, total_size):
        """下载进度回调"""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            size_mb = total_size / 1024 / 1024
            downloaded_mb = downloaded / 1024 / 1024
            print(f"\r下载进度: {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)", end="")
        else:
            downloaded_mb = downloaded / 1024 / 1024
            print(f"\r已下载: {downloaded_mb:.1f} MB", end="")
    
    def download_whisper_model(self, model_name: str) -> bool:
        """下载OpenAI Whisper模型"""
        if model_name not in self.model_configs:
            logger.error(f"未知模型: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        model_path = self.models_dir / config["filename"]
        
        if model_path.exists():
            logger.info(f"模型已存在: {model_path}")
            return True
        
        logger.info(f"正在下载 {model_name} ({config['size']})...")
        
        try:
            # 下载文件
            urllib.request.urlretrieve(
                config["url"], 
                model_path,
                reporthook=self.download_progress_hook
            )
            print()  # 换行
            
            # 验证文件完整性
            if self.verify_file_hash(model_path, config.get("sha256")):
                logger.info(f"模型下载完成: {model_path}")
                self.model_status[model_name] = {
                    "downloaded": True,
                    "path": str(model_path),
                    "size": config["size"]
                }
                self.save_status()
                return True
            else:
                logger.error("文件校验失败，删除损坏的文件")
                model_path.unlink()
                return False
                
        except Exception as e:
            logger.error(f"下载失败: {e}")
            if model_path.exists():
                model_path.unlink()
            return False
    
    def download_faster_whisper_model(self, model_name: str) -> bool:
        """下载Faster-Whisper模型"""
        if model_name not in self.model_configs:
            logger.error(f"未知模型: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        model_dir = self.models_dir / "faster-whisper" / model_name.replace("faster-whisper-", "")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if len(list(model_dir.glob("*"))) > 0:
            logger.info(f"模型已存在: {model_dir}")
            return True
        
        logger.info(f"正在下载 {model_name} ({config['size']})...")
        
        try:
            # 使用huggingface_hub下载
            try:
                from huggingface_hub import snapshot_download
                
                snapshot_download(
                    repo_id=config["hf_model"],
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                
                logger.info(f"模型下载完成: {model_dir}")
                self.model_status[model_name] = {
                    "downloaded": True,
                    "path": str(model_dir),
                    "size": config["size"]
                }
                self.save_status()
                return True
                
            except ImportError:
                logger.error("huggingface_hub未安装，无法下载Faster-Whisper模型")
                return False
                
        except Exception as e:
            logger.error(f"下载失败: {e}")
            if model_dir.exists():
                shutil.rmtree(model_dir)
            return False
    
    def download_funasr_model(self, model_name: str) -> bool:
        """下载FunASR模型"""
        if model_name not in self.model_configs:
            logger.error(f"未知模型: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        model_dir = self.models_dir / "funasr" / model_name.replace("funasr-", "")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if len(list(model_dir.glob("*"))) > 0:
            logger.info(f"模型已存在: {model_dir}")
            return True
        
        logger.info(f"正在下载 {model_name} ({config['size']})...")
        
        try:
            # 使用modelscope下载
            try:
                from modelscope import snapshot_download
                
                snapshot_download(
                    config["modelscope_model"],
                    cache_dir=str(model_dir.parent),
                    local_files_only=False
                )
                
                logger.info(f"模型下载完成: {model_dir}")
                self.model_status[model_name] = {
                    "downloaded": True,
                    "path": str(model_dir),
                    "size": config["size"]
                }
                self.save_status()
                return True
                
            except ImportError:
                logger.error("modelscope未安装，无法下载FunASR模型")
                return False
                
        except Exception as e:
            logger.error(f"下载失败: {e}")
            if model_dir.exists():
                shutil.rmtree(model_dir)
            return False
    
    def download_model(self, model_name: str) -> bool:
        """下载指定模型"""
        if self.is_model_downloaded(model_name):
            logger.info(f"模型已存在: {model_name}")
            return True
        
        if model_name.startswith("whisper-"):
            return self.download_whisper_model(model_name)
        elif model_name.startswith("faster-whisper-"):
            return self.download_faster_whisper_model(model_name)
        elif model_name.startswith("funasr-"):
            return self.download_funasr_model(model_name)
        else:
            logger.error(f"未知模型类型: {model_name}")
            return False
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """获取模型路径"""
        if not self.is_model_downloaded(model_name):
            return None
        
        config = self.model_configs.get(model_name)
        if not config:
            return None
        
        # OpenAI Whisper模型
        if "filename" in config:
            return str(self.models_dir / config["filename"])
        
        # Faster-Whisper模型
        elif "hf_model" in config:
            return str(self.models_dir / "faster-whisper" / model_name.replace("faster-whisper-", ""))
        
        # FunASR模型
        elif "modelscope_model" in config:
            return str(self.models_dir / "funasr" / model_name.replace("funasr-", ""))
        
        return None
    
    def verify_file_hash(self, file_path: Path, expected_hash: Optional[str]) -> bool:
        """验证文件哈希值"""
        if not expected_hash:
            return True  # 无需验证
        
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_hash = sha256_hash.hexdigest()
            return actual_hash == expected_hash
            
        except Exception as e:
            logger.error(f"文件验证失败: {e}")
            return False
    
    def list_models(self) -> List[Dict]:
        """列出所有可用模型"""
        models = []
        for name, config in self.model_configs.items():
            models.append({
                "name": name,
                "size": config["size"],
                "downloaded": self.is_model_downloaded(name),
                "path": self.get_model_path(name)
            })
        return models
    
    def cleanup_models(self):
        """清理损坏的模型文件"""
        logger.info("正在清理损坏的模型...")
        
        for model_name in self.model_configs:
            if model_name in self.model_status:
                path = self.model_status[model_name].get("path")
                if path and not os.path.exists(path):
                    logger.info(f"删除失效记录: {model_name}")
                    del self.model_status[model_name]
        
        self.save_status()
        logger.info("清理完成")


def main():
    """测试函数"""
    manager = ModelManager()
    
    print("可用模型:")
    for model in manager.list_models():
        status = "✅" if model["downloaded"] else "❌"
        print(f"  {status} {model['name']} ({model['size']})")
    
    # 测试下载
    test_model = "whisper-base"
    print(f"\n测试下载: {test_model}")
    if manager.download_model(test_model):
        print(f"下载成功: {manager.get_model_path(test_model)}")
    else:
        print("下载失败")


if __name__ == "__main__":
    main()
