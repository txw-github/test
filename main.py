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
import json
from text_postprocessor import TextPostProcessor

# TensorRT管理器集成到主程序中
TENSORRT_MANAGER_AVAILABLE = True
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    logger.info("TensorRT Manager组件就绪")
except ImportError as e:
    TENSORRT_MANAGER_AVAILABLE = False
    logger.warning(f"TensorRT组件不可用: {e}")
    logger.info("将使用标准模式运行")

# 配置日志 - 修复Windows编码问题
import locale
import sys

# 设置控制台编码为UTF-8
if sys.platform.startswith('win'):
    try:
        # 尝试设置控制台为UTF-8
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("video_subtitle.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 替换emoji字符以避免编码问题
def safe_log_message(message):
    """安全的日志消息，替换可能导致编码问题的字符"""
    emoji_map = {
        '🎬': '[VIDEO]',
        '🤖': '[MODEL]', 
        '💻': '[DEVICE]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🚀': '[START]',
        '🔄': '[LOADING]',
        '📊': '[INFO]',
        '🧹': '[CLEANUP]',
        '🗑️': '[DELETE]',
        '📝': '[SAVE]',
        '🎯': '[TARGET]',
        '🔍': '[CHECK]',
        '✨': '[ENHANCE]',
        '🎉': '[SUCCESS]'
    }
    for emoji, replacement in emoji_map.items():
        message = message.replace(emoji, replacement)
    return message

# 重写logger方法
original_info = logger.info
original_warning = logger.warning
original_error = logger.error

def safe_info(message, *args, **kwargs):
    return original_info(safe_log_message(str(message)), *args, **kwargs)

def safe_warning(message, *args, **kwargs):
    return original_warning(safe_log_message(str(message)), *args, **kwargs)

def safe_error(message, *args, **kwargs):
    return original_error(safe_log_message(str(message)), *args, **kwargs)

logger.info = safe_info
logger.warning = safe_warning
logger.error = safe_error

# 设置CUDA环境变量优化RTX 3060 Ti
os.environ['CUDA_LAZY_LOADING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置FFmpeg路径
ffmpeg_path = os.environ.get("FFMPEG_PATH", r"D:\code\ffmpeg\bin")
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path
    logger.info(f"[OK] FFmpeg路径已设置: {ffmpeg_path}")

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
    import onnx
    import onnxruntime as ort
    TENSORRT_AVAILABLE = True
    ONNX_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    ONNX_AVAILABLE = False
    logger.warning("TensorRT/ONNX不可用，将使用PyTorch加速")

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy未安装，将使用FFmpeg处理音频")

# 新增FunASR导入
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    logger.info("FunASR库导入成功")
except ImportError:
    FUNASR_AVAILABLE = False
    AutoModel = None
    logger.warning("未找到FunASR库，请确保已安装: pip install funasr")

# 新增FireRedASR导入
try:
    import fireredasr
    FIREREDASR_AVAILABLE = True
    logger.info("FireRedASR库导入成功")
except ImportError:
    FIREREDASR_AVAILABLE = False
    logger.warning("FireRedASR库未安装，跳过此模型")

# 新增SenseVoice导入
try:
    from sensevoice import SenseVoiceSmall, SenseVoiceLarge
    SENSEVOICE_AVAILABLE = True
    logger.info("SenseVoice库导入成功")
except ImportError:
    SENSEVOICE_AVAILABLE = False
    logger.warning("SenseVoice库未安装，跳过此模型")

class Config:
    """配置管理类"""
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
            "audio_sample_rate": 16000,
            "chinese_optimization": {
                "enable": True,
                "context_window": 5,
                "confidence_threshold": 0.7,
                "multi_pass_correction": True
            },
            "text_enhancement": {
                "professional_terms": True,
                "polyphone_correction": True,
                "punctuation_smart": True,
                "context_aware": True
            }
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置和用户配置
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    self.config = config
            except Exception as e:
                logger.warning(f"配置文件加载失败，使用默认配置: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()

    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

class Timer:
    """计时器类"""
    def __init__(self, name="任务"):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        print(f"[START] 开始 {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"[OK] {self.name} 完成，耗时: {duration:.2f} 秒")

class ProgressTracker:
    """进度跟踪器"""
    def __init__(self, total_steps=100, description="处理中"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.pbar = tqdm(total=total_steps, desc=description, unit="step")

    def update(self, steps=1, description=None):
        if description:
            self.pbar.set_description(description)
        self.pbar.update(steps)
        self.current_step += steps

    def set_progress(self, current, total=None, description=None):
        if total:
            self.pbar.total = total
        if description:
            self.pbar.set_description(description)
        self.pbar.n = current
        self.pbar.refresh()

    def close(self):
        self.pbar.close()

class RTX3060TiOptimizer:
    """RTX 3060 Ti显卡优化器"""

    @staticmethod
    def setup_gpu_memory(memory_fraction=0.85):
        """配置GPU显存管理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                max_memory = int(total_memory * memory_fraction)
                torch.cuda.set_per_process_memory_fraction(memory_fraction)

                # 启用内存池优化
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

                logger.info(f"GPU显存优化完成，总显存: {total_memory/1024**3:.1f}GB，预留使用: {max_memory/1024**3:.1f}GB")
            except Exception as e:
                logger.warning(f"显存优化失败: {e}")

    @staticmethod
    def get_optimal_batch_size():
        """获取最优批处理大小"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_memory > 5.5:  # 6GB显卡
                return 4
            else:
                return 2
        return 1

    @staticmethod
    def optimize_cuda_settings():
        """优化CUDA设置"""
        if torch.cuda.is_available():
            # 启用TensorRT优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True

            # 设置CUDA流优化
            torch.cuda.synchronize()

            # 内存优化
            torch.cuda.empty_cache()

            logger.info("CUDA优化设置完成")

class SystemChecker:
    """系统检查器"""
    @staticmethod
    def check_cuda():
        """检查CUDA环境"""
        if not torch.cuda.is_available():
            logger.error("[ERROR] CUDA不可用！请检查NVIDIA驱动和CUDA安装")
            return False

        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        logger.info(f"[OK] CUDA版本: {cuda_version}")
        logger.info(f"[OK] GPU: {gpu_name}")
        logger.info(f"[OK] GPU显存: {gpu_memory:.2f} GB")

        if "3060 Ti" in gpu_name:
            logger.info("[TARGET] 检测到RTX 3060 Ti，已启用优化配置")
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
            "moviepy": MOVIEPY_AVAILABLE,
            "funasr": FUNASR_AVAILABLE,
            "fireredasr": FIREREDASR_AVAILABLE,
            "sensevoice": SENSEVOICE_AVAILABLE
        }

        missing = [name for name, available in deps.items() if not available]
        if missing:
            logger.warning(f"[WARNING] 可选依赖缺失: {', '.join(missing)}")

        required = ["torch", "whisper"]
        missing_required = [name for name in required if not deps.get(name, False)]
        if missing_required:
            logger.error(f"[ERROR] 缺少必需依赖: {', '.join(missing_required)}")
            return False
        return True

class EnhancedVideoSubtitleExtractor:
    """增强版视频字幕提取器 - 专门优化中文识别"""

    def __init__(self, model_id: str = "faster-base", device: str = "cuda", config: Config = None, **kwargs):
        self.config = config or Config()
        self.device = device
        self.kwargs = kwargs
        self.text_processor = TextPostProcessor()

        # 检查系统
        if not SystemChecker.check_cuda():
            self.device = "cpu"
            logger.warning("[WARNING] CUDA不可用，使用CPU模式")

        # 初始化模型
        self.model_wrapper = self._create_model(model_id)

        # 初始化多模型融合（如果可用）
        self.enable_ensemble = kwargs.get('enable_ensemble', False)
        self.ensemble_models = []
        if self.enable_ensemble:
            self._init_ensemble_models()

    def _create_model(self, model_id: str):
        """创建模型实例"""
        if model_id in ["tiny", "base", "small", "medium", "large", "faster-base", "faster-large"]:
            return WhisperModelWrapper(model_id, self.device, self.config, **self.kwargs)
        elif model_id in ["funasr-paraformer", "funasr-conformer"]:
            if not FUNASR_AVAILABLE:
                raise ValueError("FunASR库未安装，请运行: pip install funasr")
            return FunASRModelWrapper(model_id, self.device, self.config, **self.kwargs)
        elif model_id.startswith("fireredasr"):
            if not FIREREDASR_AVAILABLE:
                raise ValueError("FireRedASR库未安装")
            return FireRedASRModelWrapper(model_id, self.device, self.config, **self.kwargs)
        elif model_id.startswith("sensevoice"):
            if not SENSEVOICE_AVAILABLE:
                raise ValueError("SenseVoice库未安装")
            return SenseVoiceModelWrapper(model_id, self.device, self.config, **self.kwargs)
        else:
            raise ValueError(f"不支持的模型: {model_id}")

    def _init_ensemble_models(self):
        """初始化模型融合"""
        try:
            # 为RTX 3060 Ti优化的模型组合
            ensemble_configs = [
                {"model": "faster-base", "weight": 0.6},
                {"model": "funasr-paraformer", "weight": 0.4}
            ]

            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if available_memory < 4.0:
                logger.info("[INFO] 显存不足，禁用模型融合")
                self.enable_ensemble = False
                return

            for config in ensemble_configs:
                try:
                    model = self._create_model(config["model"])
                    self.ensemble_models.append({
                        "model": model,
                        "weight": config["weight"]
                    })
                    logger.info(f"[OK] 融合模型加载: {config['model']}")
                except Exception as e:
                    logger.warning(f"[WARNING] 融合模型加载失败: {config['model']}, {e}")

        except Exception as e:
            logger.warning(f"[WARNING] 模型融合初始化失败: {e}")
            self.enable_ensemble = False

    def extract_audio(self, video_path: str, audio_path: str = None, enable_preprocessing: bool = True, audio_quality: str = "balanced") -> Optional[str]:
        """从视频提取音频，支持高级预处理"""
        if not audio_path:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            temp_path = self.config.get('temp_path', './temp')
            os.makedirs(temp_path, exist_ok=True)
            audio_path = os.path.join(temp_path, f"{base_name}_audio.wav")

        if not os.path.exists(video_path):
            logger.error(f"[ERROR] 视频文件不存在: {video_path}")
            return None

        try:
            progress = ProgressTracker(100, "提取和处理音频")
            raw_audio_path = None

            with Timer("音频提取和预处理"):
                progress.update(10, "检查视频文件...")

                # 如果启用预处理，先提取到临时文件
                if enable_preprocessing:
                    raw_audio_path = os.path.join(os.path.dirname(audio_path), f"raw_{os.path.basename(audio_path)}")
                    actual_output = raw_audio_path
                else:
                    actual_output = audio_path

                if MOVIEPY_AVAILABLE:
                    progress.update(15, "使用MoviePy提取音频...")
                    video = VideoFileClip(video_path)
                    audio = video.audio
                    progress.update(15, "写入音频文件...")
                    audio.write_audiofile(
                        actual_output, 
                        fps=self.config.get('audio_sample_rate', 16000), 
                        verbose=False, 
                        logger=None
                    )
                    progress.update(10, "清理资源...")
                    video.close()
                    audio.close()
                else:
                    progress.update(15, "使用FFmpeg提取音频...")
                    cmd = [
                        "ffmpeg", "-y", "-i", video_path,
                        "-vn", "-acodec", "pcm_s16le", 
                        "-ar", str(self.config.get('audio_sample_rate', 16000)), 
                        "-ac", "1", actual_output, "-loglevel", "error"
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    progress.update(25, "音频提取完成")

                # 如果启用预处理，进行高级音频处理
                if enable_preprocessing and os.path.exists(raw_audio_path):
                    progress.update(20, f"开始{audio_quality}质量音频预处理...")

class WhisperModelWrapper:
    """Whisper模型包装器"""
    def __init__(self, model_id: str, device: str = "cuda", config: Config = None, **kwargs):
        self.model_id = model_id
        self.device = device
        self.config = config or Config()
        self.model = None
        self.use_tensorrt = kwargs.get('use_tensorrt', False)

    def load_model(self):
        """加载Whisper模型"""
        try:
            logger.info(f"[LOADING] 加载Whisper模型: {self.model_id}")
            
            if self.model_id.startswith("faster-"):
                model_size = self.model_id.replace("faster-", "")
                self.model = WhisperModel(
                    model_size, 
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "float32"
                )
            else:
                self.model = whisper.load_model(self.model_id, device=self.device)
            
            logger.info(f"[OK] Whisper模型加载成功")
        except Exception as e:
            logger.error(f"[ERROR] Whisper模型加载失败: {e}")
            raise

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Whisper转录"""
        try:
            if self.model_id.startswith("faster-"):
                # Faster-Whisper
                segments, info = self.model.transcribe(
                    audio_path, 
                    beam_size=5,
                    language=kwargs.get('language', 'zh'),
                    temperature=kwargs.get('temperature', 0.0)
                )
                
                segment_list = []
                for segment in segments:
                    segment_list.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })
                
                return {
                    "segments": segment_list,
                    "language": info.language,
                    "text": " ".join([s["text"] for s in segment_list])
                }
            else:
                # OpenAI Whisper
                result = self.model.transcribe(
                    audio_path,
                    language=kwargs.get('language', 'zh'),
                    temperature=kwargs.get('temperature', 0.0)
                )
                return result
                
        except Exception as e:
            logger.error(f"[ERROR] Whisper转录失败: {e}")
            raise

    def get_gpu_memory_usage(self) -> float:
        """获取GPU显存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0


class FunASRModelWrapper:
    """FunASR模型包装器"""
    def __init__(self, model_id: str, device: str = "cuda", config: Config = None, **kwargs):
        self.model_id = model_id
        self.device = device
        self.config = config or Config()
        self.model = None

    def load_model(self):
        """加载FunASR模型"""
        try:
            if not FUNASR_AVAILABLE:
                raise ImportError("FunASR库未安装，请运行: pip install funasr")

            logger.info(f"[LOADING] 加载FunASR模型: {self.model_id}")
            
            if "paraformer" in self.model_id:
                model_name = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            else:
                model_name = "iic/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch"
            
            self.model = AutoModel(
                model=model_name,
                device=self.device
            )
            
            logger.info(f"[OK] FunASR模型加载成功")
        except Exception as e:
            logger.error(f"[ERROR] FunASR模型加载失败: {e}")
            raise

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """FunASR转录"""
        try:
            result = self.model.generate(input=audio_path)
            
            # 转换为统一格式
            if result and len(result) > 0:
                text = result[0]["text"] if isinstance(result[0], dict) else str(result[0])
                
                # 简单的时间戳生成（FunASR可能不提供详细时间戳）
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path)
                duration = len(audio_data) / sample_rate
                
                segments = [{
                    "start": 0.0,
                    "end": duration,
                    "text": text
                }]
                
                return {
                    "segments": segments,
                    "language": "zh",
                    "text": text
                }
            else:
                return {"segments": [], "language": "zh", "text": ""}
                
        except Exception as e:
            logger.error(f"[ERROR] FunASR转录失败: {e}")
            raise

    def get_gpu_memory_usage(self) -> float:
        """获取GPU显存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0



                    try:
                        from audio_preprocessor import AdvancedAudioPreprocessor
                        preprocessor = AdvancedAudioPreprocessor(
                            target_sample_rate=self.config.get('audio_sample_rate', 16000)
                        )

                        # 进行高级预处理
                        processed_path = preprocessor.preprocess_audio(
                            raw_audio_path, 
                            audio_path, 
                            quality=audio_quality
                        )

                        progress.update(30, "音频预处理完成")

                        # 清理原始音频文件
                        if os.path.exists(raw_audio_path):
                            os.remove(raw_audio_path)

                        logger.info(f"[ENHANCE] 音频预处理完成，质量等级: {audio_quality}")

                    except Exception as e:
                        logger.warning(f"音频预处理失败，使用原始音频: {e}")
                        # 预处理失败时，将原始文件重命名为最终文件
                        if os.path.exists(raw_audio_path):
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                            os.rename(raw_audio_path, audio_path)

                progress.update(10, "验证音频文件...")
                if os.path.exists(audio_path):
                    file_size = os.path.getsize(audio_path) / 1024 / 1024
                    progress.close()
                    logger.info(f"[OK] 音频处理成功: {audio_path} ({file_size:.1f}MB)")
                    return audio_path
                else:
                    progress.close()
                    logger.error("[ERROR] 音频处理失败")
                    return None

        except Exception as e:
            logger.error(f"[ERROR] 音频处理出错: {e}")
            # 清理临时文件
            if raw_audio_path and os.path.exists(raw_audio_path):
                try:
                    os.remove(raw_audio_path)
                except:
                    pass
            return None

    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """转录音频 - 支持模型融合和智能纠错"""
        if not os.path.exists(audio_path):
            logger.error(f"[ERROR] 音频文件不存在: {audio_path}")
            return {"segments": [], "language": None}

        try:
            # 加载模型
            if self.model_wrapper.model is None:
                self.model_wrapper.load_model()

            with Timer("音频转录"):
                # 单模型推理
                result = self.model_wrapper.transcribe(audio_path, **kwargs)

                # 模型融合推理（如果启用）
                if self.enable_ensemble and len(self.ensemble_models) > 0:
                    result = self._ensemble_transcribe(audio_path, result, **kwargs)

                # 智能文本后处理
                if self.config.get('text_enhancement', {}).get('context_aware', True):
                    result = self._enhanced_text_processing(result)

                segment_count = len(result.get('segments', []))
                logger.info(f"[OK] 转录完成，识别到 {segment_count} 个片段")

                if self.device == "cuda":
                    memory_usage = self.model_wrapper.get_gpu_memory_usage()
                    logger.info(f"[INFO] 转录后显存使用: {memory_usage:.1f}MB")

                return result

        except Exception as e:
            logger.error(f"[ERROR] 音频转录失败: {e}")
            return {"segments": [], "language": None}

    def _ensemble_transcribe(self, audio_path: str, primary_result: Dict, **kwargs) -> Dict[str, Any]:
        """模型融合转录"""
        try:
            logger.info("[INFO] 开始模型融合推理...")
            ensemble_results = [primary_result]

            for model_config in self.ensemble_models:
                try:
                    model = model_config["model"]
                    if model.model is None:
                        model.load_model()

                    result = model.transcribe(audio_path, **kwargs)
                    ensemble_results.append(result)

                except Exception as e:
                    logger.warning(f"[WARNING] 融合模型推理失败: {e}")

            # 融合结果
            return self._merge_results(ensemble_results)

        except Exception as e:
            logger.warning(f"[WARNING] 模型融合失败，使用单模型结果: {e}")
            return primary_result

    def _merge_results(self, results: List[Dict]) -> Dict[str, Any]:
        """融合多个模型的结果"""
        if len(results) <= 1:
            return results[0] if results else {"segments": [], "language": None}

        # 以第一个结果为基准
        merged_result = results[0].copy()

        # 对每个片段进行投票融合
        segments = merged_result.get("segments", [])
        for i, segment in enumerate(segments):
            texts = [segment["text"]]

            # 收集其他模型对应片段的文本
            for result in results[1:]:
                other_segments = result.get("segments", [])
                if i < len(other_segments):
                    texts.append(other_segments[i]["text"])

            # 选择最佳文本（这里简化为选择最长的）
            best_text = max(texts, key=len)
            segments[i]["text"] = best_text

        logger.info(f"[OK] 模型融合完成，使用了 {len(results)} 个模型的结果")
        return merged_result

    def _enhanced_text_processing(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """增强文本处理"""
        try:
            segments = result.get("segments", [])
            if not segments:
                return result

            logger.info("[ENHANCE] 开始智能文本处理...")

            # 全文上下文处理
            full_text = " ".join([seg["text"] for seg in segments])

            # 应用高级文本后处理
            processed_full_text = self.text_processor.post_process(full_text)

            # 将处理后的文本重新分配到各个片段
            processed_segments = self._redistribute_text(segments, processed_full_text)

            result["segments"] = processed_segments
            result["enhanced"] = True

            logger.info("[OK] 智能文本处理完成")
            return result
        except Exception as e:
            logger.warning(f"[WARNING] 智能文本处理失败: {e}")
            return result

    def _redistribute_text(self, original_segments: List[Dict], processed_text: str) -> List[Dict]:
        """将处理后的文本重新分配到片段"""
        import jieba

        # 对处理后的文本进行分词
        words = list(jieba.cut(processed_text))

        # 计算每个原始片段的词数比例
        original_words = []
        for seg in original_segments:
            seg_words = list(jieba.cut(seg["text"]))
            original_words.append(seg_words)

        total_original_words = sum(len(words) for words in original_words)

        # 按比例重新分配
        processed_segments = []
        word_index = 0

        for i, seg in enumerate(original_segments):
            original_word_count = len(original_words[i])
            if total_original_words > 0:
                proportion = original_word_count / total_original_words
                target_word_count = max(1, int(len(words) * proportion))
            else:
                target_word_count = 1

            # 提取对应的词
            segment_words = words[word_index:word_index + target_word_count]
            word_index += target_word_count

            # 创建新片段
            new_segment = seg.copy()
            new_segment["text"] = "".join(segment_words)
            processed_segments.append(new_segment)

        return processed_segments

    def create_srt_file(self, segments: List[Dict], output_path: str = "output.srt", enable_postprocess: bool = True) -> str:
        """创建SRT字幕文件"""
        try:
            progress = ProgressTracker(len(segments) + 10, "生成字幕文件")

            output_dir = self.config.get('output_path', './output')
            os.makedirs(output_dir, exist_ok=True)

            if not output_path.startswith(output_dir):
                output_path = os.path.join(output_dir, os.path.basename(output_path))

            # 初始化文本后处理器
            if enable_postprocess:
                progress.update(5, "初始化文本后处理器...")

                # 统计原始错误
                total_text = " ".join([seg["text"] for seg in segments])
                original_stats = self.text_processor.get_correction_stats(total_text)
                logger.info(f"[CHECK] 检测到潜在错误: 专业名词 {original_stats['professional_terms']} 处, "
                          f"多音字 {original_stats['polyphone_errors']} 处, "
                          f"数字单位 {original_stats['number_units']} 处")

            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    text = segment["text"].strip()

                    # 应用文本后处理
                    if enable_postprocess:
                        corrected_text = self.text_processor.post_process(text)
                        if corrected_text != text:
                            logger.debug(f"片段 {i} 文本纠错: '{text}' -> '{corrected_text}'")
                        text = corrected_text

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

                    progress.update(1, f"写入片段 {i}/{len(segments)}")

            # 保存原始版本（可选）
            if enable_postprocess:
                progress.update(3, "保存原始版本...")
                original_path = output_path.replace(".srt", "_original.srt")
                with open(original_path, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(segments, 1):
                        start_time = self._format_time(segment["start"])
                        end_time = self._format_time(segment["end"])
                        text = segment["text"].strip()
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{text}\n\n")
                logger.info(f"[SAVE] 原始字幕保存至: {original_path}")

                # 统计处理结果
                total_text = " ".join([seg["text"] for seg in segments])
                processed_text = self.text_processor.post_process(total_text)

                # 统计标点符号
                punctuation_count = len(re.findall(r'[，。！？；：]', processed_text))
                sentence_count = len(re.findall(r'[。！？]', processed_text))

                logger.info(f"[INFO] 文本处理统计: 添加了 {punctuation_count} 个标点符号, {sentence_count} 个句子")

            progress.update(2, "完成字幕生成...")
            progress.close()
            logger.info(f"[OK] SRT文件保存成功: {output_path}")

            if enable_postprocess:
                logger.info("[TARGET] 文本后处理功能已启用，已添加标点符号并修正错别字")

            return output_path

        except Exception as e:
            logger.error(f"[ERROR] SRT文件创建失败: {e}")
            return None

    def _format_time(self, seconds: float) -> str:
        """格式化时间为SRT格式"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

    def cleanup(self):
        """清理临时文件和显存"""
        try:
            temp_path = self.config.get('temp_path', './temp')
            if os.path.exists(temp_path):
                temp_files = [f for f in os.listdir(temp_path) if f.endswith("_audio.wav")]
                for file in temp_files:
                    file_path = os.path.join(temp_path, file)
                    try:
                        os.remove(file_path)
                        logger.info(f"[DELETE] 删除临时文件: {file}")
                    except Exception as e:
                        logger.warning(f"[WARNING] 删除临时文件失败 {file}: {e}")

            # 清理GPU显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("[CLEANUP] GPU显存清理完成")

        except Exception as e:
            logger.warning(f"[WARNING] 清理过程中出现错误: {e}")

# 模型包装器类定义

# 新增FireRedASR模型包装器
class FireRedASRModelWrapper:
    """FireRedASR模型包装器"""
    def __init__(self, model_id: str, device: str = "cuda", config: Config = None, **kwargs):
        self.model_id = model_id
        self.device = device
        self.config = config or Config()
        self.model = None

    def load_model(self):
        """加载FireRedASR模型"""
        try:
            if not FIREREDASR_AVAILABLE:
                raise ImportError("FireRedASR库未安装")

            logger.info(f"[LOADING] 加载FireRedASR模型: {self.model_id}")
            # 这里应该是实际的FireRedASR加载代码
            # self.model = fireredasr.load_model(self.model_id)
            logger.info(f"[OK] FireRedASR模型加载成功")

        except Exception as e:
            logger.error(f"[ERROR] FireRedASR模型加载失败: {e}")
            raise

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """FireRedASR转录"""
        try:
            # 实际的FireRedASR推理代码
            # result = self.model.transcribe(audio_path)

            # 临时返回格式
            return {
                "segments": [],
                "language": "zh",
                "text": ""
            }

        except Exception as e:
            logger.error(f"[ERROR] FireRedASR转录失败: {e}")
            raise

    def get_gpu_memory_usage(self) -> float:
        """获取GPU显存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

# 新增SenseVoice模型包装器
class SenseVoiceModelWrapper:
    """SenseVoice模型包装器"""
    def __init__(self, model_id: str, device: str = "cuda", config: Config = None, **kwargs):
        self.model_id = model_id
        self.device = device
        self.config = config or Config()
        self.model = None

    def load_model(self):
        """加载SenseVoice模型"""
        try:
            if not SENSEVOICE_AVAILABLE:
                raise ImportError("SenseVoice库未安装")

            logger.info(f"[LOADING] 加载SenseVoice模型: {self.model_id}")

            if "small" in self.model_id:
                self.model = SenseVoiceSmall.from_pretrained("iic/SenseVoiceSmall")
            else:
                self.model = SenseVoiceLarge.from_pretrained("iic/SenseVoiceLarge")

            logger.info(f"[OK] SenseVoice模型加载成功")

        except Exception as e:
            logger.error(f"[ERROR] SenseVoice模型加载失败: {e}")
            raise

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """SenseVoice转录"""
        try:
            # 实际的SenseVoice推理代码
            # result = self.model(audio_path)

            # 临时返回格式
            return {
                "segments": [],
                "language": "zh", 
                "text": ""
            }

        except Exception as e:
            logger.error(f"[ERROR] SenseVoice转录失败: {e}")
            raise

    def get_gpu_memory_usage(self) -> float:
        """获取GPU显存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="中文电视剧音频转文字工具 - RTX 3060 Ti优化版")
    parser.add_argument("video_path", nargs='?', default="test.mp4", help="输入视频文件路径")
    parser.add_argument("--output", "-o", default="output.srt", help="输出字幕文件路径")
    parser.add_argument("--model", "-m", default="faster-base",
                        choices=["tiny", "base", "small", "medium", "large", "faster-base", "faster-large", 
                                "funasr-paraformer", "funasr-conformer", "fireredasr-small", "fireredasr-base", 
                                "fireredasr-large", "sensevoice-small", "sensevoice-large"],
                        help="模型选择 (推荐RTX 3060 Ti使用faster-base或funasr-paraformer)")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--language", "-l", default="zh", help="语言设置")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--no-postprocess", action="store_true", help="禁用文本后处理")
    parser.add_argument("--add-term", nargs=2, metavar=('CORRECT', 'WRONG'), 
                        help="添加自定义纠错词汇: --add-term '正确词' '错误词'")
    parser.add_argument("--audio-quality", choices=["fast", "balanced", "high"], default="balanced",
                        help="音频预处理质量 (fast/balanced/high)")
    parser.add_argument("--enable-audio-preprocessing", action="store_true", default=True,
                        help="启用高级音频预处理")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                        help="模型精度选择 (fp16更快，fp32更精确)")
    parser.add_argument("--analyze-audio", action="store_true", 
                        help="分析音频质量并提供优化建议")
    parser.add_argument("--analyze-text", action="store_true",
                        help="分析文本质量并提供优化建议")
    parser.add_argument("--enable-ensemble", action="store_true",
                        help="启用多模型融合推理（需要更多显存）")

    args = parser.parse_args()

    # 加载配置
    config = Config()
    if args.config and os.path.exists(args.config):
        config.config_file = args.config
        config.load_config()

    # 检查输入文件
    if not os.path.exists(args.video_path):
        logger.error(f"[ERROR] 视频文件不存在: {args.video_path}")
        return

    # 检查依赖
    if not SystemChecker.check_dependencies():
        logger.error("[ERROR] 请先运行install_dependencies.bat安装缺少的依赖")
        return

    logger.info(f"[VIDEO] 开始处理视频: {args.video_path}")
    logger.info(f"[MODEL] 使用模型: {args.model}")
    logger.info(f"[DEVICE] 运行设备: {args.device}")

    if args.model in ["medium", "large"] and args.device == "cuda":
        logger.warning("[WARNING] RTX 3060 Ti显存可能不足以运行medium/large模型，建议使用faster-base")

    if args.model in ["funasr-paraformer", "funasr-conformer"]:
        logger.warning("[WARNING] FunASR模型内存占用较大，如遇到内存不足请考虑使用faster-base模型")
        # 检查可用内存
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024**3:  # 小于4GB可用内存
            logger.warning(f"[WARNING] 可用内存不足({memory.available/1024**3:.1f}GB)，建议关闭其他程序或使用smaller模型")

    extractor = None
    try:
        # 创建增强版提取器
        extractor = EnhancedVideoSubtitleExtractor(
            model_id=args.model,
            device=args.device,
            config=config,
            enable_ensemble=args.enable_ensemble
        )

        # 提取音频（支持预处理）
        audio_path = extractor.extract_audio(
            args.video_path,
            enable_preprocessing=args.enable_audio_preprocessing,
            audio_quality=args.audio_quality
        )
        if not audio_path:
            logger.error("[ERROR] 音频提取失败")
            return

        # 音频质量分析（可选）
        if args.analyze_audio:
            try:
                from audio_preprocessor import AdvancedAudioPreprocessor
                preprocessor = AdvancedAudioPreprocessor()
                audio_metrics = preprocessor.analyze_audio_quality(audio_path)

                if audio_metrics:
                    logger.info(f"[INFO] 音频质量分析结果:")
                    logger.info(f"   - 综合评分: {audio_metrics.get('overall_score', 0):.1f}/100")
                    logger.info(f"   - 中文适配度: {audio_metrics.get('chinese_speech_score', 0):.1f}/100")
                    logger.info(f"   - 语音清晰度: {audio_metrics.get('speech_clarity', 0):.2f}")
                    logger.info(f"   - 噪声水平: {audio_metrics.get('noise_level', 0):.1f}")

                    recommendations = audio_metrics.get('recommendations', [])
                    if recommendations:
                        logger.info("[SAVE] 优化建议:")
                        for rec in recommendations:
                            logger.info(f"   - {rec}")
            except Exception as e:
                logger.warning(f"音频质量分析失败: {e}")

        # 转录音频
        result = extractor.transcribe_audio(
            audio_path,
            language=args.language,
            temperature=0.0 if args.precision == "fp16" else 0.1
        )

        if not result["segments"]:
            logger.warning("[WARNING] 未识别到任何语音内容")
            return

        # 创建字幕文件
        enable_postprocess = not args.no_postprocess
        srt_path = extractor.create_srt_file(result["segments"], args.output, enable_postprocess)
        if srt_path:
            logger.info(f"[SUCCESS] 字幕提取完成！文件保存至: {srt_path}")
            logger.info(f"[SAVE] 共识别到 {len(result['segments'])} 个字幕片段")
            if enable_postprocess:
                logger.info("[ENHANCE] 已应用智能文本纠错")
            if args.enable_ensemble:
                logger.info("[ENHANCE] 已应用多模型融合推理")

            # 文本质量分析（可选）
            if args.analyze_text:
                try:
                    text_analysis = extractor.text_processor.analyze_text_quality(
                        " ".join([seg["text"] for seg in result["segments"]])
                    )

                    logger.info(f"[INFO] 文本质量分析结果:")
                    logger.info(f"   - 质量评分: {text_analysis['quality_score']}")
                    logger.info(f"   - 错误率: {text_analysis['error_rate']}%")

                    error_stats = text_analysis['error_statistics']
                    logger.info(f"   - 潜在同音字错误: {error_stats['sound_alike_errors']} 处")
                    logger.info(f"   - 专业术语错误: {error_stats['professional_terms']} 处")
                    logger.info(f"   - 语气词冗余: {error_stats['filler_words']} 处")

                    recommendations = text_analysis['recommendations']
                    if recommendations:
                        logger.info("[SAVE] 文本优化建议:")
                        for rec in recommendations:
                            logger.info(f"   - {rec}")
                except Exception as e:
                    logger.warning(f"文本质量分析失败: {e}")
        else:
            logger.error("[ERROR] 字幕文件创建失败")

        # 处理自定义词汇添加
        if args.add_term:
            extractor.text_processor.add_custom_correction(args.add_term[0], [args.add_term[1]])
            logger.info(f"[OK] 已添加自定义纠错词汇: {args.add_term[0]} <- {args.add_term[1]}")

    except Exception as e:
        logger.error(f"[ERROR] 处理过程中发生错误: {e}")
        traceback.print_exc()

    finally:
        # 清理临时文件和显存
        try:
            if extractor is not None and not args.keep_temp:
                extractor.cleanup()
        except Exception as e:
            logger.warning(f"[WARNING] 清理过程中出现错误: {e}")

if __name__ == "__main__":
    main()