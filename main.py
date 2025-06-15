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
from audio_preprocessor import AdvancedAudioPreprocessor
from model_manager import ModelManager

# 设置FFmpeg路径
os.environ["PATH"] += os.pathsep + r"D:\code\ffmpeg\bin"

# TensorRT和优化库支持
TENSORRT_AVAILABLE = True
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    logger.info("TensorRT优化引擎就绪")
except ImportError as e:
    TENSORRT_AVAILABLE = False
    logger.warning(f"TensorRT不可用: {e}")

# 配置日志系统
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

# CUDA优化设置
os.environ['CUDA_LAZY_LOADING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 导入各种ASR库
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
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    logger.info("FunASR库导入成功")
except ImportError:
    FUNASR_AVAILABLE = False
    logger.warning("FunASR未安装")

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy未安装，将使用FFmpeg")

class EnhancedConfig:
    """增强配置管理"""
    def __init__(self):
        self.config_file = "enhanced_config.json"
        self.load_config()

    def load_config(self):
        default_config = {
            "models": {
                "preferred_models": {
                    "RTX 3060 Ti": ["faster-base", "funasr-paraformer", "whisper-base"],
                    "RTX 3060": ["faster-base", "whisper-small"],
                    "RTX 3070": ["faster-large", "whisper-medium", "funasr-conformer"]
                },
                "precision_levels": {
                    "high": {"fp16": True, "batch_size": 1, "beam_size": 5},
                    "balanced": {"fp16": True, "batch_size": 2, "beam_size": 3},
                    "fast": {"fp16": False, "batch_size": 4, "beam_size": 1}
                }
            },
            "audio": {
                "advanced_preprocessing": True,
                "denoise_strength": 0.7,
                "voice_enhancement": True,
                "chinese_optimization": True,
                "sample_rate": 16000
            },
            "text": {
                "postprocessing": True,
                "professional_terms": True,
                "polyphone_correction": True,
                "punctuation_smart": True,
                "context_aware": True
            },
            "optimization": {
                "tensorrt_enabled": True,
                "multi_model_ensemble": False,
                "memory_optimization": True,
                "gpu_memory_fraction": 0.8
            },
            "paths": {
                "models_path": "./models",
                "temp_path": "./temp",
                "output_path": "./output",
                "cache_path": "./cache"
            }
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 深度合并配置
                    self.config = self._deep_merge(default_config, config)
            except Exception as e:
                logger.warning(f"配置加载失败: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()

    def _deep_merge(self, base, update):
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"配置保存失败: {e}")

    def get(self, path, default=None):
        """支持路径访问：config.get('models.precision_levels.high')"""
        keys = path.split('.')
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

class MultiModelWrapper:
    """多模型包装器 - 支持模型切换和集成"""
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.models = {}
        self.current_model = None
        self.model_manager = ModelManager()

    def load_optimal_model(self, precision="balanced"):
        """加载最优模型"""
        try:
            # 获取推荐模型
            optimal_model = self.model_manager.get_optimal_model()
            logger.info(f"选择最优模型: {optimal_model}")

            # 加载模型
            if optimal_model.startswith("faster-"):
                self.current_model = self._load_faster_whisper(optimal_model, precision)
            elif optimal_model.startswith("funasr"):
                self.current_model = self._load_funasr(optimal_model, precision)
            elif optimal_model.startswith("whisper"):
                self.current_model = self._load_whisper(optimal_model, precision)
            else:
                # 降级到基础模型
                self.current_model = self._load_faster_whisper("faster-base", precision)

            return self.current_model is not None

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def _load_faster_whisper(self, model_name, precision):
        """加载Faster-Whisper模型"""
        try:
            size = model_name.replace("faster-", "")
            precision_config = self.config.get(f"models.precision_levels.{precision}", {})

            model = WhisperModel(
                size,
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float16" if precision_config.get("fp16", True) else "float32",
                cpu_threads=4,
                download_root=self.config.get("paths.models_path")
            )

            logger.info(f"Faster-Whisper {size} 模型加载成功")
            return {"model": model, "type": "faster-whisper", "precision": precision_config}

        except Exception as e:
            logger.error(f"Faster-Whisper加载失败: {e}")
            return None

    def _load_funasr(self, model_name, precision):
        """加载FunASR模型"""
        try:
            if not FUNASR_AVAILABLE:
                return None

            model_mapping = {
                "funasr-paraformer": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                "funasr-conformer": "damo/speech_conformer_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            }

            actual_model = model_mapping.get(model_name, model_mapping["funasr-paraformer"])
            precision_config = self.config.get(f"models.precision_levels.{precision}", {})

            model = AutoModel(
                model=actual_model,
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_dir=self.config.get("paths.models_path"),
                disable_update=True,
                batch_size=precision_config.get("batch_size", 1)
            )

            logger.info(f"FunASR {model_name} 模型加载成功")
            return {"model": model, "type": "funasr", "precision": precision_config}

        except Exception as e:
            logger.error(f"FunASR加载失败: {e}")
            return None

    def _load_whisper(self, model_name, precision):
        """加载标准Whisper模型"""
        try:
            size = model_name.replace("whisper-", "")
            model = whisper.load_model(
                size, 
                download_root=self.config.get("paths.models_path")
            )

            if torch.cuda.is_available():
                model = model.cuda()

            precision_config = self.config.get(f"models.precision_levels.{precision}", {})

            logger.info(f"Whisper {size} 模型加载成功")
            return {"model": model, "type": "whisper", "precision": precision_config}

        except Exception as e:
            logger.error(f"Whisper加载失败: {e}")
            return None

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """智能转录"""
        if not self.current_model:
            raise ValueError("未加载任何模型")

        model_info = self.current_model
        model_type = model_info["type"]
        model = model_info["model"]
        precision_config = model_info["precision"]

        try:
            if model_type == "faster-whisper":
                return self._transcribe_faster_whisper(model, audio_path, precision_config, **kwargs)
            elif model_type == "funasr":
                return self._transcribe_funasr(model, audio_path, precision_config, **kwargs)
            elif model_type == "whisper":
                return self._transcribe_whisper(model, audio_path, precision_config, **kwargs)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

        except Exception as e:
            logger.error(f"转录失败: {e}")
            return {"segments": [], "language": "zh", "error": str(e)}

    def _transcribe_faster_whisper(self, model, audio_path, precision_config, **kwargs):
        """Faster-Whisper转录"""
        segments, info = model.transcribe(
            audio_path,
            language="zh",
            beam_size=precision_config.get("beam_size", 3),
            best_of=precision_config.get("best_of", 3),
            temperature=kwargs.get("temperature", 0.0),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        result = {"text": "", "segments": [], "language": info.language}

        for segment in segments:
            result["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            result["text"] += segment.text.strip() + " "

        return result

    def _transcribe_funasr(self, model, audio_path, precision_config, **kwargs):
        """FunASR转录"""
        result_list = model.generate(
            input=audio_path,
            cache={},
            language="zh",
            use_itn=True,
            batch_size_s=60,
            batch_size=precision_config.get("batch_size", 1)
        )

        result = {"text": "", "segments": [], "language": "zh"}

        for i, res in enumerate(result_list):
            text = res.get("text", "")
            if text:
                start_time = i * 30.0
                end_time = (i + 1) * 30.0

                result["segments"].append({
                    "start": start_time,
                    "end": end_time,
                    "text": text.strip()
                })
                result["text"] += text.strip() + " "

        return result

    def _transcribe_whisper(self, model, audio_path, precision_config, **kwargs):
        """标准Whisper转录"""
        result = model.transcribe(
            audio_path,
            language="zh",
            fp16=precision_config.get("fp16", True),
            verbose=False,
            temperature=kwargs.get("temperature", 0.0)
        )
        return result

class EnhancedVideoSubtitleExtractor:
    """增强版视频字幕提取器"""
    def __init__(self, config: EnhancedConfig = None, precision="balanced"):
        self.config = config or EnhancedConfig()
        self.precision = precision

        # 初始化组件
        self.audio_preprocessor = AdvancedAudioPreprocessor(
            config_path="audio_config.json"
        )
        self.text_postprocessor = TextPostProcessor(
            config_file="text_correction_config.json"
        )
        self.model_wrapper = MultiModelWrapper(self.config)

        # 加载最优模型
        if not self.model_wrapper.load_optimal_model(precision):
            logger.error("模型加载失败")
            raise RuntimeError("无法加载任何可用模型")

    def extract_and_enhance_audio(self, video_path: str) -> Optional[str]:
        """提取并增强音频"""
        try:
            # 第一步：基础音频提取
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            temp_path = self.config.get("paths.temp_path", "./temp")
            os.makedirs(temp_path, exist_ok=True)

            raw_audio_path = os.path.join(temp_path, f"{base_name}_raw.wav")
            enhanced_audio_path = os.path.join(temp_path, f"{base_name}_enhanced.wav")

            logger.info("🎵 开始音频提取和增强...")

            # 使用FFmpeg提取音频
            extract_cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(self.config.get("audio.sample_rate", 16000)),
                "-ac", "1", "-loglevel", "error",
                raw_audio_path
            ]

            subprocess.run(extract_cmd, check=True, capture_output=True)

            if not os.path.exists(raw_audio_path):
                logger.error("原始音频提取失败")
                return None

            # 第二步：高级音频预处理
            if self.config.get("audio.advanced_preprocessing", True):
                logger.info("🔧 执行高级音频预处理...")
                processed_path = self.audio_preprocessor.preprocess_audio(
                    raw_audio_path, enhanced_audio_path
                )

                if processed_path and os.path.exists(processed_path):
                    # 音频质量分析
                    quality_metrics = self.audio_preprocessor.analyze_audio_quality(processed_path)
                    logger.info(f"📊 音频质量评分: {quality_metrics.get('overall_score', 0):.1f}/100")

                    # 清理原始文件
                    try:
                        os.remove(raw_audio_path)
                    except:
                        pass

                    return processed_path
                else:
                    logger.warning("音频预处理失败，使用原始音频")
                    return raw_audio_path
            else:
                return raw_audio_path

        except Exception as e:
            logger.error(f"音频提取增强失败: {e}")
            return None

    def transcribe_with_enhancement(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """增强转录"""
        try:
            logger.info("🎯 开始增强转录...")

            # 执行转录
            result = self.model_wrapper.transcribe(audio_path, **kwargs)

            if not result["segments"]:
                logger.warning("未识别到任何语音内容")
                return result

            # 统计原始结果
            total_segments = len(result["segments"])
            total_text = " ".join([seg["text"] for seg in result["segments"]])

            logger.info(f"📝 原始转录: {total_segments} 个片段, {len(total_text)} 个字符")

            # 文本后处理增强
            if self.config.get("text.postprocessing", True):
                logger.info("✨ 执行智能文本后处理...")

                # 获取错误统计
                error_stats = self.text_postprocessor.get_correction_stats(total_text)
                logger.info(f"🔍 检测到潜在错误: 专业名词 {error_stats['professional_terms']}, "
                          f"多音字 {error_stats['polyphone_errors']}, "
                          f"同音字 {error_stats['sound_alike_errors']}")

                # 批量处理所有片段
                enhanced_segments = []
                corrections_made = 0

                for segment in result["segments"]:
                    original_text = segment["text"]
                    enhanced_text = self.text_postprocessor.post_process(original_text)

                    if enhanced_text != original_text:
                        corrections_made += 1
                        logger.debug(f"文本纠错: '{original_text}' -> '{enhanced_text}'")

                    enhanced_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": enhanced_text,
                        "original_text": original_text
                    })

                result["segments"] = enhanced_segments
                result["text"] = " ".join([seg["text"] for seg in enhanced_segments])
                result["corrections_made"] = corrections_made

                logger.info(f"✅ 文本后处理完成，修正了 {corrections_made} 处错误")

            return result

        except Exception as e:
            logger.error(f"增强转录失败: {e}")
            return {"segments": [], "language": "zh", "error": str(e)}

    def create_enhanced_srt(self, segments: List[Dict], output_path: str) -> str:
        """创建增强SRT文件"""
        try:
            output_dir = self.config.get("paths.output_path", "./output")
            os.makedirs(output_dir, exist_ok=True)

            if not output_path.startswith(output_dir):
                output_path = os.path.join(output_dir, os.path.basename(output_path))

            # 创建主SRT文件
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    text = segment["text"].strip()

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

            # 创建对比文件（包含原始文本）
            if any("original_text" in seg for seg in segments):
                comparison_path = output_path.replace(".srt", "_comparison.srt")
                with open(comparison_path, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(segments, 1):
                        start_time = self._format_time(segment["start"])
                        end_time = self._format_time(segment["end"])
                        enhanced_text = segment["text"].strip()
                        original_text = segment.get("original_text", enhanced_text)

                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"原始: {original_text}\n")
                        f.write(f"优化: {enhanced_text}\n\n")

                logger.info(f"📊 对比文件已保存: {comparison_path}")

            logger.info(f"✅ SRT文件保存成功: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"SRT文件创建失败: {e}")
            return None

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

    def cleanup(self):
        """清理资源"""
        try:
            # 清理临时文件
            temp_path = self.config.get("paths.temp_path", "./temp")
            if os.path.exists(temp_path):
                temp_files = [f for f in os.listdir(temp_path) 
                             if f.endswith(('.wav', '.tmp'))]
                for file in temp_files:
                    try:
                        os.remove(os.path.join(temp_path, file))
                    except:
                        pass

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            logger.info("🧹 资源清理完成")

        except Exception as e:
            logger.warning(f"清理过程出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="增强版中文视频字幕提取工具")
    parser.add_argument("video_path", nargs='?', default="test.mp4", help="输入视频文件")
    parser.add_argument("--output", "-o", default="output.srt", help="输出字幕文件")
    parser.add_argument("--precision", "-p", default="balanced", 
                       choices=["high", "balanced", "fast"], help="精度级别")
    parser.add_argument("--model", "-m", help="指定模型")
    parser.add_argument("--no-enhance", action="store_true", help="禁用音频增强")
    parser.add_argument("--no-postprocess", action="store_true", help="禁用文本后处理")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.video_path):
        logger.error(f"❌ 视频文件不存在: {args.video_path}")
        return

    logger.info(f"🎬 开始处理视频: {args.video_path}")
    logger.info(f"🔧 精度级别: {args.precision}")

    extractor = None
    try:
        # 创建配置
        config = EnhancedConfig()

        # 根据参数调整配置
        if args.no_enhance:
            config.config["audio"]["advanced_preprocessing"] = False
        if args.no_postprocess:
            config.config["text"]["postprocessing"] = False

        # 创建提取器
        extractor = EnhancedVideoSubtitleExtractor(config, args.precision)

        # 提取和增强音频
        audio_path = extractor.extract_and_enhance_audio(args.video_path)
        if not audio_path:
            logger.error("❌ 音频处理失败")
            return

        # 增强转录
        result = extractor.transcribe_with_enhancement(audio_path)
        if not result["segments"]:
            logger.error("❌ 转录失败或无语音内容")
            return

        # 创建增强SRT
        srt_path = extractor.create_enhanced_srt(result["segments"], args.output)
        if srt_path:
            logger.info(f"🎉 转换完成！")
            logger.info(f"📁 输出文件: {srt_path}")
            logger.info(f"📊 识别片段: {len(result['segments'])}")
            if "corrections_made" in result:
                logger.info(f"✨ 文本修正: {result['corrections_made']} 处")

    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        traceback.print_exc()

    finally:
        if extractor and not args.keep_temp:
            extractor.cleanup()

if __name__ == "__main__":
    main()