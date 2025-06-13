
import os
import sys
import time
import torch
import argparse
import traceback
import subprocess
import platform
from datetime import timedelta
from typing import List, Dict, Any, Optional
import logging
import re
import numpy as np
import soundfile as sf
import gc
from tqdm import tqdm
import psutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("video_subtitle.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置CUDA环境变量
os.environ['CUDA_LAZY_LOADING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 尝试导入依赖
try:
    import whisper
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    from modelscope.models.audio.asr import FireRedAsr
    FIRERED_AVAILABLE = True
except ImportError:
    FIRERED_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

class Timer:
    """计时器类"""
    def __init__(self, name="任务"):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        print(f"开始 {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"{self.name} 完成，耗时: {duration:.2f} 秒")

class SystemChecker:
    """系统检查器"""
    @staticmethod
    def check_cuda():
        """检查CUDA环境"""
        if not torch.cuda.is_available():
            logger.error("CUDA不可用！请检查NVIDIA驱动和CUDA安装")
            return False
        
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logger.info(f"CUDA版本: {cuda_version}")
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU显存: {gpu_memory:.2f} GB")
        
        return True
    
    @staticmethod
    def check_dependencies():
        """检查依赖是否安装"""
        deps = {
            "whisper": WHISPER_AVAILABLE,
            "transformers": HF_AVAILABLE,
            "tensorrt": TENSORRT_AVAILABLE,
            "firered": FIRERED_AVAILABLE,
            "moviepy": MOVIEPY_AVAILABLE
        }
        
        missing = [name for name, available in deps.items() if not available]
        if missing:
            logger.warning(f"缺少依赖: {', '.join(missing)}")
            return False
        return True

class ModelWrapper:
    """模型包装基类"""
    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        self.model_id = model_id
        self.device = device
        self.kwargs = kwargs
        self.model = None
    
    def load_model(self):
        raise NotImplementedError
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
    
    def get_gpu_memory_usage(self) -> float:
        """获取GPU显存使用量（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

class WhisperModelWrapper(ModelWrapper):
    """Whisper模型包装"""
    def load_model(self):
        logger.info(f"加载Whisper模型: {self.model_id}")
        if "faster" in self.model_id:
            # 使用faster-whisper
            self.model = WhisperModel(
                self.model_id.replace("faster-", ""),
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "float32",
                download_root=self.kwargs.get("download_root", "models")
            )
        else:
            # 使用原版whisper
            self.model = whisper.load_model(
                self.model_id,
                device=self.device,
                download_root=self.kwargs.get("download_root", "models")
            )
        logger.info("Whisper模型加载完成")
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        try:
            if hasattr(self.model, 'transcribe'):
                if "faster" in self.model_id:
                    segments, info = self.model.transcribe(audio_path, **kwargs)
                    return {"segments": list(segments), "language": info.language}
                else:
                    result = self.model.transcribe(audio_path, **kwargs)
                    return result
            return {"segments": [], "language": None}
        except Exception as e:
            logger.error(f"Whisper转录失败: {e}")
            return {"segments": [], "language": None}

class FireRedModelWrapper(ModelWrapper):
    """FireRedASR模型包装"""
    def load_model(self):
        logger.info(f"加载FireRedASR模型: {self.model_id}")
        model_type = self.model_id.split("-")[-1] if "-" in self.model_id else "aed"
        self.model = FireRedAsr.from_pretrained(f"pengzhendong/FireRedASR-{model_type.upper()}-L")
        if self.device == "cuda":
            self.model = self.model.cuda()
        logger.info("FireRedASR模型加载完成")
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        try:
            result = self.model.transcribe(audio_path)
            # 转换为标准格式
            duration = self.get_audio_duration(audio_path)
            segments = [{
                "start": 0.0,
                "end": duration,
                "text": result[0]["text"] if result else ""
            }]
            return {"segments": segments, "language": "zh"}
        except Exception as e:
            logger.error(f"FireRedASR转录失败: {e}")
            return {"segments": [], "language": "zh"}
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import librosa
            return librosa.get_duration(path=audio_path)
        except:
            return 0.0

class ModelFactory:
    """模型工厂"""
    @staticmethod
    def create_model(model_id: str, device: str = "cuda", **kwargs) -> ModelWrapper:
        if "whisper" in model_id.lower():
            return WhisperModelWrapper(model_id, device, **kwargs)
        elif "firered" in model_id.lower():
            return FireRedModelWrapper(model_id, device, **kwargs)
        else:
            raise ValueError(f"不支持的模型: {model_id}")

class VideoSubtitleExtractor:
    """视频字幕提取器"""
    def __init__(self, model_id: str = "base", device: str = "cuda", **kwargs):
        self.device = device
        self.kwargs = kwargs
        
        # 检查系统
        if not SystemChecker.check_cuda():
            self.device = "cpu"
            logger.warning("CUDA不可用，使用CPU模式")
        
        # 初始化模型
        self.model_wrapper = ModelFactory.create_model(model_id, device=self.device, **kwargs)
        self.model_wrapper.load_model()
    
    def extract_audio(self, video_path: str, audio_path: str = None) -> Optional[str]:
        """从视频提取音频"""
        if not audio_path:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = f"{base_name}_audio.wav"
        
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return None
        
        try:
            with Timer("音频提取"):
                if MOVIEPY_AVAILABLE:
                    # 使用moviepy
                    video = VideoFileClip(video_path)
                    audio = video.audio
                    audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)
                    video.close()
                    audio.close()
                else:
                    # 使用ffmpeg
                    cmd = [
                        "ffmpeg", "-y", "-i", video_path,
                        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                        audio_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
            
            if os.path.exists(audio_path):
                logger.info(f"音频提取成功: {audio_path}")
                return audio_path
            else:
                logger.error("音频提取失败")
                return None
                
        except Exception as e:
            logger.error(f"音频提取出错: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """转录音频"""
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return {"segments": [], "language": None}
        
        try:
            with Timer("音频转录"):
                result = self.model_wrapper.transcribe(audio_path, **kwargs)
                logger.info(f"转录完成，识别到 {len(result.get('segments', []))} 个片段")
                return result
        except Exception as e:
            logger.error(f"音频转录失败: {e}")
            return {"segments": [], "language": None}
    
    def create_srt_file(self, segments: List[Dict], output_path: str = "output.srt") -> str:
        """创建SRT字幕文件"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            
            logger.info(f"SRT文件保存成功: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"SRT文件创建失败: {e}")
            return None
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间为SRT格式"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def cleanup(self):
        """清理临时文件"""
        temp_files = [f for f in os.listdir(".") if f.endswith("_audio.wav")]
        for file in temp_files:
            try:
                os.remove(file)
                logger.info(f"删除临时文件: {file}")
            except Exception as e:
                logger.warning(f"删除临时文件失败 {file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="视频字幕提取工具 - 支持多模型")
    parser.add_argument("video_path", nargs='?', default="test.mp4", help="输入视频文件路径")
    parser.add_argument("--output", "-o", default="output.srt", help="输出字幕文件路径")
    parser.add_argument("--model", "-m", default="base",
                        choices=["tiny", "base", "small", "medium", "large", "faster-base", "faster-large", "firered-aed", "firered-llm"],
                        help="模型选择")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--language", "-l", default="zh", help="语言设置")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.video_path):
        logger.error(f"视频文件不存在: {args.video_path}")
        return
    
    # 检查依赖
    if not SystemChecker.check_dependencies():
        logger.error("请先安装缺少的依赖")
        return
    
    logger.info(f"开始处理视频: {args.video_path}")
    logger.info(f"使用模型: {args.model}")
    logger.info(f"运行设备: {args.device}")
    
    try:
        # 创建提取器
        extractor = VideoSubtitleExtractor(
            model_id=args.model,
            device=args.device,
            download_root="models"
        )
        
        # 提取音频
        audio_path = extractor.extract_audio(args.video_path)
        if not audio_path:
            logger.error("音频提取失败")
            return
        
        # 转录音频
        result = extractor.transcribe_audio(
            audio_path,
            language=args.language,
            temperature=0.2
        )
        
        if not result["segments"]:
            logger.warning("未识别到任何语音内容")
            return
        
        # 创建字幕文件
        srt_path = extractor.create_srt_file(result["segments"], args.output)
        if srt_path:
            logger.info(f"字幕提取完成！文件保存至: {srt_path}")
        else:
            logger.error("字幕文件创建失败")
    
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        if not args.keep_temp:
            extractor.cleanup()

if __name__ == "__main__":
    main()
