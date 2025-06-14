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

# 设置CUDA环境变量优化RTX 3060 Ti
os.environ['CUDA_LAZY_LOADING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 尝试导入依赖
try:
    import whisper
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper库未安装")

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Transformers库未安装")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT不可用，将使用PyTorch加速")

# 暂时禁用ModelScope以避免版本冲突
try:
    from modelscope.models.audio.asr import FireRedAsr
    FIRERED_AVAILABLE = True
    # FIRERED_AVAILABLE = False
    # logger.warning("ModelScope暂时禁用以避免版本冲突，FireRedASR功能暂不可用")
except ImportError:
    FIRERED_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy未安装，将使用FFmpeg处理音频")

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

class RTX3060TiOptimizer:
    """RTX 3060 Ti显卡优化器"""

    @staticmethod
    def setup_gpu_memory():
        """配置GPU显存管理"""
        if torch.cuda.is_available():
            # 设置显存增长策略
            torch.cuda.empty_cache()
            # 预留一些显存给系统
            try:
                # 获取总显存
                total_memory = torch.cuda.get_device_properties(0).total_memory
                # RTX 3060 Ti有6GB显存，预留1GB给系统
                max_memory = int(total_memory * 0.85)  # 使用85%显存
                torch.cuda.set_per_process_memory_fraction(0.85)
                logger.info(f"GPU显存优化完成，总显存: {total_memory/1024**3:.1f}GB，预留使用: {max_memory/1024**3:.1f}GB")
            except Exception as e:
                logger.warning(f"显存优化失败: {e}")

    @staticmethod
    def get_optimal_batch_size():
        """获取最优批处理大小"""
        if torch.cuda.is_available():
            # RTX 3060 Ti 6GB显存的推荐设置
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_memory > 5.5:  # 6GB显卡
                return 4
            else:
                return 2
        return 1

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

        # 特别检查RTX 3060 Ti
        if "3060 Ti" in gpu_name:
            logger.info("检测到RTX 3060 Ti，已启用优化配置")
            RTX3060TiOptimizer.setup_gpu_memory()

        return True

    @staticmethod
    def check_dependencies():
        """检查依赖是否安装"""
        deps = {
            "torch": True,
            "whisper": WHISPER_AVAILABLE,
            "transformers": HF_AVAILABLE,
            "tensorrt": TENSORRT_AVAILABLE,
            "moviepy": MOVIEPY_AVAILABLE
        }

        missing = [name for name, available in deps.items() if not available]
        if missing:
            logger.warning(f"可选依赖缺失: {', '.join(missing)}")

        # 检查必需依赖
        required = ["torch", "whisper"]
        missing_required = [name for name in required if not deps.get(name, False)]
        if missing_required:
            logger.error(f"缺少必需依赖: {', '.join(missing_required)}")
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

        # 根据RTX 3060 Ti优化设置
        batch_size = RTX3060TiOptimizer.get_optimal_batch_size()

        if "faster" in self.model_id:
            # 使用faster-whisper，RTX 3060 Ti优化设置
            model_size = self.model_id.replace("faster-", "")
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "float32",
                download_root=self.kwargs.get("download_root", "models"),
                cpu_threads=4,
                num_workers=1,  # RTX 3060 Ti单GPU优化
            )
        else:
            # 使用原版whisper
            self.model = whisper.load_model(
                self.model_id,
                device=self.device,
                download_root=self.kwargs.get("download_root", "models")
            )

        logger.info("Whisper模型加载完成")
        if self.device == "cuda":
            logger.info(f"当前显存使用: {self.get_gpu_memory_usage():.1f}MB")

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        try:
            # RTX 3060 Ti优化参数
            if "faster" in self.model_id:
                # Faster-Whisper设置
                segments, info = self.model.transcribe(
                    audio_path, 
                    beam_size=5,  # 减少beam size以节省显存
                    best_of=5,
                    temperature=0.0,
                    **kwargs
                )
                return {"segments": list(segments), "language": info.language}
            else:
                # 原版Whisper设置
                result = self.model.transcribe(
                    audio_path, 
                    temperature=0.0,
                    **kwargs
                )
                return result
        except torch.cuda.OutOfMemoryError:
            logger.warning("显存不足，尝试释放显存后重试...")
            torch.cuda.empty_cache()
            gc.collect()
            # 使用更保守的设置重试
            if "faster" in self.model_id:
                segments, info = self.model.transcribe(
                    audio_path, 
                    beam_size=1,  # 最小beam size
                    **kwargs
                )
                return {"segments": list(segments), "language": info.language}
            else:
                result = self.model.transcribe(audio_path, **kwargs)
                return result
        except Exception as e:
            logger.error(f"Whisper转录失败: {e}")
            return {"segments": [], "language": None}

class FireRedModelWrapper(ModelWrapper):
    """FireRedASR模型包装"""
    def load_model(self):
        if not FIRERED_AVAILABLE:
            raise ImportError("FireRedASR不可用，请安装modelscope")

        logger.info(f"加载FireRedASR模型: {self.model_id}")
        # 解析模型类型：firered-aed 或 firered-llm
        model_type = self.model_id.split("-")[-1] if "-" in self.model_id else "aed"

        try:
            # 使用ModelScope的预训练模型
            if model_type.lower() == "aed":
                model_name = "pengzhendong/FireRedASR-AED-L"
            elif model_type.lower() == "llm":
                model_name = "pengzhendong/FireRedASR-LLM-L"
            else:
                model_name = "pengzhendong/FireRedASR-AED-L"  # 默认使用AED

            self.model = FireRedAsr.from_pretrained(model_name)
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()

            logger.info("FireRedASR模型加载完成")
            if self.device == "cuda":
                logger.info(f"当前显存使用: {self.get_gpu_memory_usage():.1f}MB")

        except Exception as e:
            logger.error(f"FireRedASR模型加载失败: {e}")
            raise

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return {"segments": [], "language": "zh"}

        try:
            # FireRedASR转录
            result = self.model.transcribe(
                batch_uttid=["single"],
                batch_wav_path=[audio_path],
                args={
                    "gpu": 1 if self.device == "cuda" else 0,
                    "compute_type": "float16" if self.device == "cuda" else "float32"
                }
            )

            # 获取音频时长
            duration = self.get_audio_duration(audio_path)

            # 转换为标准格式
            segments = []
            if result and len(result) > 0:
                text = result[0].get("text", "").strip()
                if text:
                    segments = [{
                        "start": 0.0,
                        "end": duration,
                        "text": text
                    }]

            return {"segments": segments, "language": "zh"}

        except torch.cuda.OutOfMemoryError:
            logger.warning("显存不足，尝试释放显存后重试...")
            torch.cuda.empty_cache()
            gc.collect()
            # 使用CPU重试
            try:
                result = self.model.cpu().transcribe(
                    batch_uttid=["single"],
                    batch_wav_path=[audio_path],
                    args={"gpu": 0, "compute_type": "float32"}
                )
                duration = self.get_audio_duration(audio_path)
                segments = []
                if result and len(result) > 0:
                    text = result[0].get("text", "").strip()
                    if text:
                        segments = [{"start": 0.0, "end": duration, "text": text}]
                return {"segments": segments, "language": "zh"}
            except Exception as e:
                logger.error(f"CPU模式FireRedASR转录也失败: {e}")
                return {"segments": [], "language": "zh"}
        except Exception as e:
            logger.error(f"FireRedASR转录失败: {e}")
            return {"segments": [], "language": "zh"}

    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import librosa
            return librosa.get_duration(path=audio_path)
        except ImportError:
            try:
                import soundfile as sf
                data, samplerate = sf.read(audio_path)
                return len(data) / samplerate
            except:
                logger.warning("无法获取音频时长，使用默认值")
                return 60.0  # 默认60秒
        except Exception as e:
            logger.warning(f"获取音频时长失败: {e}")
            return 60.0

class ModelFactory:
    """模型工厂"""
    @staticmethod
    def create_model(model_id: str, device: str = "cuda", **kwargs) -> ModelWrapper:
        if "whisper" in model_id.lower() or model_id in ["tiny", "base", "small", "medium", "large"]:
            return WhisperModelWrapper(model_id, device, **kwargs)
        elif "firered" in model_id.lower():
            return FireRedModelWrapper(model_id, device, **kwargs)
        else:
            raise ValueError(f"不支持的模型: {model_id}。支持的模型: tiny, base, small, medium, large, faster-base, faster-large, firered-aed, firered-llm")

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
                    result = subprocess.run(cmd, check=True, capture_output=True)
                    if result.returncode != 0:
                        logger.error("FFmpeg音频提取失败")
                        return None

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
                segment_count = len(result.get('segments', []))
                logger.info(f"转录完成，识别到 {segment_count} 个片段")
                if self.device == "cuda":
                    logger.info(f"转录后显存使用: {self.model_wrapper.get_gpu_memory_usage():.1f}MB")
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
        """清理临时文件和显存"""
        # 清理临时文件
        temp_files = [f for f in os.listdir(".") if f.endswith("_audio.wav")]
        for file in temp_files:
            try:
                os.remove(file)
                logger.info(f"删除临时文件: {file}")
            except Exception as e:
                logger.warning(f"删除临时文件失败 {file}: {e}")

        # 清理GPU显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def main():
    parser = argparse.ArgumentParser(description="视频字幕提取工具 - RTX 3060 Ti优化版")
    parser.add_argument("video_path", nargs='?', default="test.mp4", help="输入视频文件路径")
    parser.add_argument("--output", "-o", default="output.srt", help="输出字幕文件路径")
    parser.add_argument("--model", "-m", default="faster-base",
                        choices=["tiny", "base", "small", "medium", "large", "faster-base", "faster-large", "firered-aed", "firered-llm"],
                        help="模型选择 (推荐RTX 3060 Ti使用faster-base)")
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
        logger.error("请先运行install_dependencies.bat安装缺少的依赖")
        return

    logger.info(f"开始处理视频: {args.video_path}")
    logger.info(f"使用模型: {args.model}")
    logger.info(f"运行设备: {args.device}")

    if args.model in ["medium", "large"] and args.device == "cuda":
        logger.warning("RTX 3060 Ti显存可能不足以运行medium/large模型，建议使用faster-base")

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
            temperature=0.0
        )

        if not result["segments"]:
            logger.warning("未识别到任何语音内容")
            return

        # 创建字幕文件
        srt_path = extractor.create_srt_file(result["segments"], args.output)
        if srt_path:
            logger.info(f"字幕提取完成！文件保存至: {srt_path}")
            logger.info(f"共识别到 {len(result['segments'])} 个字幕片段")
        else:
            logger.error("字幕文件创建失败")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        traceback.print_exc()

    finally:
        # 清理临时文件和显存
        if not args.keep_temp:
            extractor.cleanup()

if __name__ == "__main__":
    main()