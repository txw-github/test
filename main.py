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

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy未安装，将使用FFmpeg处理音频")

# 新增FireRedASR导入
try:
    from funasr import FireRedAsr
    FIREREDASR_AVAILABLE = True
    logger.info("FireRedASR库导入成功")
except ImportError:
    try:
        # 尝试其他可能的导入方式
        from fireredasr import FireRedAsr
        FIREREDASR_AVAILABLE = True
        logger.info("FireRedASR库导入成功")
    except ImportError:
        FIREREDASR_AVAILABLE = False
        FireRedAsr = None
        logger.warning("未找到FireRedASR库，请确保已安装: pip install funasr 或 pip install fireredasr")

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
            "audio_sample_rate": 16000
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
        print(f"🚀 开始 {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"✅ {self.name} 完成，耗时: {duration:.2f} 秒")

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

class SystemChecker:
    """系统检查器"""
    @staticmethod
    def check_cuda():
        """检查CUDA环境"""
        if not torch.cuda.is_available():
            logger.error("❌ CUDA不可用！请检查NVIDIA驱动和CUDA安装")
            return False

        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        logger.info(f"✅ CUDA版本: {cuda_version}")
        logger.info(f"✅ GPU: {gpu_name}")
        logger.info(f"✅ GPU显存: {gpu_memory:.2f} GB")

        if "3060 Ti" in gpu_name:
            logger.info("🎯 检测到RTX 3060 Ti，已启用优化配置")
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
            logger.warning(f"⚠️ 可选依赖缺失: {', '.join(missing)}")

        required = ["torch", "whisper"]
        missing_required = [name for name in required if not deps.get(name, False)]
        if missing_required:
            logger.error(f"❌ 缺少必需依赖: {', '.join(missing_required)}")
            return False
        return True

class ModelWrapper:
    """模型包装基类"""
    def __init__(self, model_id: str, device: str = "cuda", config: Config = None, **kwargs):
        self.model_id = model_id
        self.device = device
        self.config = config or Config()
        self.kwargs = kwargs
        self.model = None
        self.progress_tracker = None

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
    def load_model(self) -> None:
        """加载模型"""
        try:
            self.progress_tracker = ProgressTracker(100, f"加载{self.model_id}模型")

            if self.device == "cuda" and torch.cuda.is_available():
                RTX3060TiOptimizer.setup_gpu_memory(self.config.get('gpu_memory_fraction', 0.85))

            models_path = self.config.get('models_path', './models')
            os.makedirs(models_path, exist_ok=True)

            self.progress_tracker.update(20, "下载模型文件...")

            if self.model_id in ["faster-base", "faster-large"]:
                model_mapping = {
                    "faster-base": "base",
                    "faster-large": "large"
                }
                actual_model = model_mapping[self.model_id]
                logger.info(f"🔄 加载Faster-Whisper模型: {self.model_id} -> {actual_model}")

                self.progress_tracker.update(30, "初始化Faster-Whisper...")
                self.model = WhisperModel(
                    actual_model,
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "int8",
                    cpu_threads=4,
                    download_root=models_path
                )
            else:
                logger.info(f"🔄 加载标准Whisper模型: {self.model_id}")
                self.progress_tracker.update(30, "初始化Whisper...")
                import whisper
                self.model = whisper.load_model(self.model_id, download_root=models_path)

                if self.device == "cuda":
                    self.model = self.model.cuda()

            self.progress_tracker.update(50, "模型加载完成")
            self.progress_tracker.close()
            logger.info(f"✅ 模型 {self.model_id} 加载成功")

        except Exception as e:
            if self.progress_tracker:
                self.progress_tracker.close()
            logger.error(f"❌ 模型加载失败: {e}")
            raise

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """转录音频"""
        try:
            progress = ProgressTracker(100, "音频转录中")

            if self.model_id in ["faster-base", "faster-large"]:
                progress.update(10, "开始Faster-Whisper转录...")
                segments, info = self.model.transcribe(
                    audio_path,
                    language="zh",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                progress.update(60, "处理转录结果...")
                result = {
                    "text": "",
                    "segments": [],
                    "language": info.language
                }

                for i, segment in enumerate(segments):
                    result["segments"].append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })
                    result["text"] += segment.text.strip() + " "
                    if i % 10 == 0:
                        progress.update(2, f"处理片段 {i+1}")

            else:
                progress.update(10, "开始标准Whisper转录...")
                result = self.model.transcribe(
                    audio_path,
                    language="zh",
                    fp16=torch.cuda.is_available(),
                    verbose=False
                )
                progress.update(80, "转录完成")

            progress.close()
            return result

        except Exception as e:
            logger.error(f"❌ 转录失败: {e}")
            raise

class VideoSubtitleExtractor:
    """视频字幕提取器"""
    def __init__(self, model_id: str = "faster-base", device: str = "cuda", config: Config = None, **kwargs):
        self.config = config or Config()
        self.device = device
        self.kwargs = kwargs

        # 检查系统
        if not SystemChecker.check_cuda():
            self.device = "cpu"
            logger.warning("⚠️ CUDA不可用，使用CPU模式")

        # 初始化模型
        self.model_wrapper = self._create_model(model_id)

    def _create_model(self, model_id: str):
        """创建模型实例"""
        if model_id in ["tiny", "base", "small", "medium", "large", "faster-base", "faster-large"]:
            return WhisperModelWrapper(model_id, self.device, self.config, **self.kwargs)
        else:
            raise ValueError(f"不支持的模型: {model_id}")

    def extract_audio(self, video_path: str, audio_path: str = None) -> Optional[str]:
        """从视频提取音频"""
        if not audio_path:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            temp_path = self.config.get('temp_path', './temp')
            os.makedirs(temp_path, exist_ok=True)
            audio_path = os.path.join(temp_path, f"{base_name}_audio.wav")

        if not os.path.exists(video_path):
            logger.error(f"❌ 视频文件不存在: {video_path}")
            return None

        try:
            progress = ProgressTracker(100, "提取音频")

            with Timer("音频提取"):
                progress.update(10, "检查视频文件...")

                if MOVIEPY_AVAILABLE:
                    progress.update(20, "使用MoviePy提取音频...")
                    video = VideoFileClip(video_path)
                    audio = video.audio
                    progress.update(30, "写入音频文件...")
                    audio.write_audiofile(
                        audio_path, 
                        fps=self.config.get('audio_sample_rate', 16000), 
                        verbose=False, 
                        logger=None
                    )
                    progress.update(30, "清理资源...")
                    video.close()
                    audio.close()
                else:
                    progress.update(20, "使用FFmpeg提取音频...")
                    cmd = [
                        "ffmpeg", "-y", "-i", video_path,
                        "-vn", "-acodec", "pcm_s16le", 
                        "-ar", str(self.config.get('audio_sample_rate', 16000)), 
                        "-ac", "1", audio_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    progress.update(60, "音频提取完成")

                progress.update(10, "验证音频文件...")
                if os.path.exists(audio_path):
                    file_size = os.path.getsize(audio_path) / 1024 / 1024
                    progress.close()
                    logger.info(f"✅ 音频提取成功: {audio_path} ({file_size:.1f}MB)")
                    return audio_path
                else:
                    progress.close()
                    logger.error("❌ 音频提取失败")
                    return None

        except Exception as e:
            logger.error(f"❌ 音频提取出错: {e}")
            return None

    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """转录音频"""
        if not os.path.exists(audio_path):
            logger.error(f"❌ 音频文件不存在: {audio_path}")
            return {"segments": [], "language": None}

        try:
            # 加载模型
            if self.model_wrapper.model is None:
                self.model_wrapper.load_model()

            with Timer("音频转录"):
                result = self.model_wrapper.transcribe(audio_path, **kwargs)
                segment_count = len(result.get('segments', []))
                logger.info(f"✅ 转录完成，识别到 {segment_count} 个片段")

                if self.device == "cuda":
                    memory_usage = self.model_wrapper.get_gpu_memory_usage()
                    logger.info(f"📊 转录后显存使用: {memory_usage:.1f}MB")

                return result

        except Exception as e:
            logger.error(f"❌ 音频转录失败: {e}")
            return {"segments": [], "language": None}

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
                postprocessor = TextPostProcessor()

                # 统计原始错误
                total_text = " ".join([seg["text"] for seg in segments])
                original_stats = postprocessor.get_correction_stats(total_text)
                logger.info(f"🔍 检测到潜在错误: 专业名词 {original_stats['professional_terms']} 处, "
                          f"多音字 {original_stats['polyphone_errors']} 处, "
                          f"数字单位 {original_stats['number_units']} 处")

            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    text = segment["text"].strip()

                    # 应用文本后处理
                    if enable_postprocess:
                        corrected_text = postprocessor.post_process(text)
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
                logger.info(f"📝 原始字幕保存至: {original_path}")

            progress.update(2, "完成字幕生成...")
            progress.close()
            logger.info(f"✅ SRT文件保存成功: {output_path}")

            if enable_postprocess:
                logger.info("🎯 文本后处理功能已启用，专业名词和多音字错误已自动修正")

            return output_path

        except Exception as e:
            logger.error(f"❌ SRT文件创建失败: {e}")
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
                        logger.info(f"🗑️ 删除临时文件: {file}")
                    except Exception as e:
                        logger.warning(f"⚠️ 删除临时文件失败 {file}: {e}")

            # 清理GPU显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("🧹 GPU显存清理完成")

        except Exception as e:
            logger.warning(f"⚠️ 清理过程中出现错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="中文电视剧音频转文字工具 - RTX 3060 Ti优化版")
    parser.add_argument("video_path", nargs='?', default="test.mp4", help="输入视频文件路径")
    parser.add_argument("--output", "-o", default="output.srt", help="输出字幕文件路径")
    parser.add_argument("--model", "-m", default="faster-base",
                        choices=["tiny", "base", "small", "medium", "large", "faster-base", "faster-large"],
                        help="模型选择 (推荐RTX 3060 Ti使用faster-base)")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--language", "-l", default="zh", help="语言设置")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--no-postprocess", action="store_true", help="禁用文本后处理")
    parser.add_argument("--add-term", nargs=2, metavar=('CORRECT', 'WRONG'), 
                        help="添加自定义纠错词汇: --add-term '正确词' '错误词'")

    args = parser.parse_args()

    # 加载配置
    config = Config()
    if args.config and os.path.exists(args.config):
        config.config_file = args.config
        config.load_config()

    # 检查输入文件
    if not os.path.exists(args.video_path):
        logger.error(f"❌ 视频文件不存在: {args.video_path}")
        return

    # 检查依赖
    if not SystemChecker.check_dependencies():
        logger.error("❌ 请先运行install_dependencies.bat安装缺少的依赖")
        return

    logger.info(f"🎬 开始处理视频: {args.video_path}")
    logger.info(f"🤖 使用模型: {args.model}")
    logger.info(f"💻 运行设备: {args.device}")

    if args.model in ["medium", "large"] and args.device == "cuda":
        logger.warning("⚠️ RTX 3060 Ti显存可能不足以运行medium/large模型，建议使用faster-base")

    extractor = None
    try:
        # 创建提取器
        extractor = VideoSubtitleExtractor(
            model_id=args.model,
            device=args.device,
            config=config
        )

        # 提取音频
        audio_path = extractor.extract_audio(args.video_path)
        if not audio_path:
            logger.error("❌ 音频提取失败")
            return

        # 转录音频
        result = extractor.transcribe_audio(
            audio_path,
            language=args.language,
            temperature=0.0
        )

        if not result["segments"]:
            logger.warning("⚠️ 未识别到任何语音内容")
            return

        # 创建字幕文件
        enable_postprocess = not args.no_postprocess
        srt_path = extractor.create_srt_file(result["segments"], args.output, enable_postprocess)
        if srt_path:
            logger.info(f"🎉 字幕提取完成！文件保存至: {srt_path}")
            logger.info(f"📝 共识别到 {len(result['segments'])} 个字幕片段")
            if enable_postprocess:
                logger.info("✨ 已应用智能文本纠错")
        else:
            logger.error("❌ 字幕文件创建失败")

        # 处理自定义词汇添加
        if args.add_term:
            from text_postprocessor import TextPostProcessor
            postprocessor = TextPostProcessor()
            postprocessor.add_custom_term(args.add_term[0], [args.add_term[1]])
            logger.info(f"✅ 已添加自定义纠错词汇: {args.add_term[0]} <- {args.add_term[1]}")

    except Exception as e:
        logger.error(f"❌ 处理过程中发生错误: {e}")
        traceback.print_exc()

    finally:
        # 清理临时文件和显存
        try:
            if extractor is not None and not args.keep_temp:
                extractor.cleanup()
        except Exception as e:
            logger.warning(f"⚠️ 清理过程中出现错误: {e}")

if __name__ == "__main__":
    main()