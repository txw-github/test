import os
import time
import torch
import argparse
import traceback
import subprocess
import platform
from datetime import timedelta
from typing import List, Dict, Any, Optional
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import ctypes
import psutil
import logging
import sys
import re
import jieba
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

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

# 设置FFmpeg路径（根据您的实际路径修改）
os.environ["PATH"] += os.pathsep + r"D:\code\ffmpeg\bin"

# 添加CUDA路径到系统PATH
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
if os.path.exists(cuda_path) and cuda_path not in os.environ["PATH"]:
    os.environ["PATH"] = cuda_path + ";" + os.environ["PATH"]
    logger.info(f"已将CUDA路径添加到系统PATH: {cuda_path}")


# 模型包装基类
class ModelWrapper:
    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.kwargs = kwargs
        self.model = None
        self.processor = None

    def load_model(self):
        """加载模型抽象方法"""
        raise NotImplementedError

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """转录抽象方法"""
        raise NotImplementedError


# Whisper模型包装类
class WhisperModelWrapper(ModelWrapper):
    def load_model(self):
        model_size = self.model_id.replace("whisper-","")  # 解析model_id格式如whisper-large-v3
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.kwargs.get("compute_type", "int8_float16"),
            download_root=self.kwargs.get("download_root", "models"),
            cpu_threads=self.kwargs.get("cpu_threads", 4),
        )
        logger.info(f"Whisper模型 {self.model_id} 加载成功")

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        try:
            segments, info = self.model.transcribe(audio_path, **kwargs)
            return {"segments": segments, "language": info.language}
        except Exception as e:
            logger.error(f"Whisper转录失败: {e}")
            traceback.print_exc()
            return {"segments": [], "language": None}


# Hugging Face模型包装类（改进版）
class HuggingFaceModelWrapper(ModelWrapper):
    def load_model(self):
        logger.info(f"加载Hugging Face模型: {self.model_id}")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=self.kwargs.get("download_root", "models")
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.kwargs.get("download_root", "models")
        )

        # 创建pipeline用于ASR任务
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=4096,  # 增加最大生成长度
            chunk_length_s=30,  # 30秒分块处理
            batch_size=4,  # 批处理大小
            torch_dtype=torch_dtype,
            device=self.device,
        )

        logger.info(f"Hugging Face模型 {self.model_id} 加载完成")

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        try:
            # 设置默认参数
            transcription_args = {
                "return_timestamps": True,
                "generate_kwargs": {
                    "temperature": kwargs.get("temperature", 0.2),
                    "num_beams": kwargs.get("beam_size", 5),
                }
            }

            # 新增：强制中文解码（针对BELLE等模型）
            if "belle-whisper" in self.model_id or kwargs.get("language") == "zh":
                if hasattr(self.processor, "get_decoder_prompt_ids"):
                    # 获取Whisper的中文解码提示ID
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language="zh", task="transcribe"
                    )
                    transcription_args["generate_kwargs"]["forced_decoder_ids"] = forced_decoder_ids
                    logger.info("已启用中文强制解码")

            # 语言设置
            if kwargs.get("language"):
                transcription_args["generate_kwargs"]["language"] = kwargs["language"]

            # 处理长音频时的特殊参数
            if self.model_id.startswith("openai/whisper") or "belle-whisper" in self.model_id:
                transcription_args["generate_kwargs"]["task"] = "transcribe"

            # 执行转录
            result = self.pipe(audio_path, **transcription_args)

            # 格式化输出为与faster-whisper兼容的格式
            segments = []
            if "chunks" in result:  # 带时间戳的结果
                for chunk in result["chunks"]:
                    segments.append({
                        "start": chunk["timestamp"][0],
                        "end": chunk["timestamp"][1],
                        "text": chunk["text"]
                    })
            else:  # 只有文本的结果
                segments.append({
                    "start": 0.0,
                    "end": self.get_audio_duration(audio_path),
                    "text": result["text"]
                })

            return {"segments": segments, "language": kwargs.get("language", "unknown")}

        except Exception as e:
            logger.error(f"Hugging Face模型转录失败: {e}")
            traceback.print_exc()
            return {"segments": [], "language": None}

    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            return duration
        except:
            # 回退方案
            command = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            return float(result.stdout.strip()) if result.stdout else 0.0


# 模型工厂类（更新）
class ModelFactory:
    MODEL_MAPPING = {
        "whisper": WhisperModelWrapper,  # faster-whisper
        "hf-whisper": HuggingFaceModelWrapper,  # Hugging Face Whisper系列
        "hf-belle-whisper": HuggingFaceModelWrapper,  # BELLE等第三方Whisper变体
        "hf": HuggingFaceModelWrapper,  # 通用Hugging Face ASR模型
    }

    @staticmethod
    def create_model(model_id: str, **kwargs) -> ModelWrapper:
        # 新增对BELLE等自定义Whisper模型的检测
        if model_id.startswith("hf-belle-") or "belle-whisper" in model_id:
            return ModelFactory.MODEL_MAPPING["hf-belle-whisper"](model_id, **kwargs)
        elif model_id.startswith("hf-"):
            model_type = model_id.split("-")[1]
            full_type = f"hf-{model_type}"
            if full_type in ModelFactory.MODEL_MAPPING:
                return ModelFactory.MODEL_MAPPING[full_type](model_id, **kwargs)
            else:
                return ModelFactory.MODEL_MAPPING["hf"](model_id, **kwargs)
        elif 'whisper' in model_id:
            return ModelFactory.MODEL_MAPPING["whisper"](model_id, **kwargs)
        else:
            raise ValueError(f"不支持的模型格式: {model_id}")


class Timer:
    """计时器类，用于性能分析"""

    def __init__(self, name="任务"):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"开始 {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        logger.info(f"{self.name} 完成，耗时: {duration:.2f} 秒")
        return False


class SystemMonitor:
    """系统资源监控器"""

    @staticmethod
    def get_system_info():
        info = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "gpu_usage": 0,
            "gpu_memory": 0
        }
        if torch.cuda.is_available():
            try:
                info["gpu_usage"] = torch.cuda.utilization(0)
                info["gpu_memory"] = torch.cuda.memory_allocated(0) / (1024 ** 3)
            except:
                pass
        return info

    @staticmethod
    def log_system_info():
        info = SystemMonitor.get_system_info()
        logger.info(
            f"系统资源: CPU={info['cpu_usage']}%, 内存={info['memory_usage']}%, "
            f"GPU={info['gpu_usage']}%, 显存={info['gpu_memory']:.2f}GB"
        )

    @staticmethod
    def check_environment():
        logger.info("=" * 50)
        logger.info("环境检测开始...")
        logger.info(f"系统: {platform.system()} {platform.release()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__} {'(CUDA)' if torch.cuda.is_available() else '(CPU)'}")
        if torch.cuda.is_available():
            logger.info(
                f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB)")
        try:
            ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], text=True).split('\n')[0]
            logger.info(f"FFmpeg: {ffmpeg_version}")
        except:
            logger.error("FFmpeg不可用！")
        logger.info("环境检测完成")
        logger.info("=" * 50)


class AudioPreprocessor:
    @staticmethod
    def normalize_audio(audio_path, output_path):
        try:
            data, rate = sf.read(audio_path)
            meter = pyln.Meter(rate)
            loudness = meter.integrated_loudness(data)
            normalized = pyln.normalize.loudness(data, loudness, -16.0)
            sf.write(output_path, normalized, rate, 'WAV', 'PCM_16')
            logger.info(f"音量标准化完成 (原始:{loudness:.1f} LUFS -> 目标:-16 LUFS)")
            return output_path
        except:
            return AudioPreprocessor.ffmpeg_normalize(audio_path, output_path)

    @staticmethod
    def ffmpeg_normalize(audio_path, output_path):
        command = [
            "ffmpeg", "-y", "-i", audio_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            "-loglevel", "error", output_path
        ]
        subprocess.run(command)
        logger.info("FFmpeg音量标准化完成")
        return output_path

    @staticmethod
    def denoise_audio(audio_path, output_path):
        command = [
            "ffmpeg", "-y", "-i", audio_path,
            "-af", "arnndn=m=rnnoise_models/somnolent-hogwash.rnnn",
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            "-loglevel", "error", output_path
        ]
        subprocess.run(command)
        logger.info("降噪处理完成")
        return output_path


class ChineseTextProcessor:
    @staticmethod
    def enhance_chinese_text(text):
        text = re.sub(r'\s+', '', text)
        corrections = {
            "甚么": "什么", "哪裡": "哪里", "後来": "后来",
            "好把": "好吧", "好巴": "好吧", "好伐": "好吗"
        }
        for w, c in corrections.items():
            text = text.replace(w, c)
        if text and text[-1] not in '。！？':
            text += '。'
        return ' '.join(jieba.cut(text))


# 视频字幕提取器类（更新）
class VideoSubtitleExtractor:
    def __init__(self, model_id: str = "whisper-large-v3", device: str = "cuda", **kwargs):
        self.device = device
        self.model_id = model_id
        self.kwargs = kwargs
        SystemMonitor.check_environment()

        # 针对BELLE模型调整默认参数
        if "belle-whisper" in model_id:
            self.kwargs.update({
                "beam_size": 15,  # 增加波束搜索宽度提升中文准确率
                "temperature": 0.1  # 降低温度减少随机性
            })
            logger.info(f"BELLE模型参数优化: beam_size=15, temperature=0.1")

        # 初始化模型
        self.model_wrapper = ModelFactory.create_model(model_id, device=device, **kwargs)
        self.model_wrapper.load_model()

        # GPU内存限制
        if device == "cuda" and torch.cuda.is_available():
            self.set_gpu_memory_limit(kwargs.get("max_gpu_memory", 4500))

    def set_gpu_memory_limit(self, limit_mb):
        if self.device != "cuda":
            return
        try:
            cudart = ctypes.CDLL('cudart64_12.dll')
            MB = 1024 * 1024
            cudart.cudaDeviceSetLimit(0x05, limit_mb * MB)
            logger.info(f"设置GPU显存限制: {limit_mb}MB")
        except:
            logger.warning("GPU显存限制设置失败")

    def extract_audio(self, video_path):
        audio_path = "extracted_audio.wav"
        command = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "48000",
            "-acodec", "pcm_s24le", "-loglevel", "error",
            audio_path
        ]
        subprocess.run(command)
        logger.info(f"音频提取完成: {audio_path}")
        return audio_path

    def preprocess_audio(self, audio_path, denoise=True):
        processed_path = "processed_audio.wav"
        temp_path = "temp_audio.wav"

        # 音量标准化
        normalized_path = AudioPreprocessor.normalize_audio(audio_path, temp_path)

        # 降噪处理
        if denoise:
            denoised_path = AudioPreprocessor.denoise_audio(normalized_path, processed_path)
        else:
            denoised_path = normalized_path

        # 格式转换
        command = [
            "ffmpeg", "-y", "-i", denoised_path,
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            "-loglevel", "error", processed_path
        ]
        subprocess.run(command)
        return processed_path

    def transcribe_audio(self, audio_path, language=None, temperature=0.2):
        options = {
            "language": language,
            "temperature": temperature,
            "vad_filter": True,
            "beam_size": 10,
            "initial_prompt": "以下是普通话对话，包含日常用语和专有名词。"
        }
        result = self.model_wrapper.transcribe(audio_path, **options)

        # 中文文本处理
        if result["language"] == "zh":
            for seg in result["segments"]:
                seg["text"] = ChineseTextProcessor.enhance_chinese_text(seg["text"])
        return result

    def create_srt_file(self, segments, output_path="output.srt"):
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                # 处理不同类型的segment对象
                if hasattr(seg, 'start'):  # 处理faster-whisper的Segment类
                    start = seg.start
                    end = seg.end
                    text = seg.text
                else:  # 处理字典类型的segment
                    start = seg["start"]
                    end = seg["end"]
                    text = seg["text"]

                start_time = self._format_time(start)
                end_time = self._format_time(end)
                text = text.replace('\n', ' ').strip()
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
        logger.info(f"SRT文件生成完成: {output_path}")
        return output_path

    def _format_time(self, seconds):
        td = timedelta(seconds=seconds)
        return f"{td.seconds // 3600:02d}:" \
               f"{(td.seconds // 60) % 60:02d}:" \
               f"{td.seconds % 60:02d},{int((seconds - int(seconds)) * 1000):03d}"

    def cleanup(self, temp_files):
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        logger.info("临时文件清理完成")


def main():
    parser = argparse.ArgumentParser(
        description="多模型支持的视频字幕提取工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("video_path", nargs='?', default="1.mkv", help="输入视频文件路径")
    # parser.add_argument("--model-id", default="whisper-large-v1",help="模型标识（格式：{type}-{name}，如whisper-large-v3或hf-whisper-tiny）")
    parser.add_argument("--model-id", default="BELLE-2/Belle-whisper-large-v3-zh-punct",help="模型标识（格式：{type}-{name}，如whisper-large-v3或hf-whisper-tiny）")
    parser.add_argument("--output", "-o", default="output.srt", help="输出字幕文件路径")
    parser.add_argument("--device", "-d", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--language", "-l", default=None, help="强制语言代码（如zh/en）")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--no-denoise", action="store_true", help="禁用降噪")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")
    parser.add_argument("--max-gpu-memory", type=int, default=4500, help="最大GPU显存(MB)")

    args = parser.parse_args()

    # 设备自动检测
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {args.device}")

    extractor = VideoSubtitleExtractor(
        model_id=args.model_id,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        compute_type="int8_float16",
        download_root="models"
    )

    temp_files = []
    try:
        with Timer("音频提取"):
            audio_path = extractor.extract_audio(args.video_path)
            temp_files.append(audio_path)

        with Timer("音频预处理"):
            processed_audio = extractor.preprocess_audio(audio_path, denoise=not args.no_denoise)
            temp_files.append(processed_audio)

        with Timer("语音转录"):
            result = extractor.transcribe_audio(processed_audio, args.language, args.temperature)
            if not result["segments"]:
                logger.error("未检测到语音内容")
                return

        with Timer("字幕生成"):
            extractor.create_srt_file(result["segments"], args.output)

    except Exception as e:
        logger.error(f"处理失败: {e}")
        traceback.print_exc()
    finally:
        if not args.keep_temp:
            extractor.cleanup(temp_files)
        extractor.model_wrapper.model = None  # 释放模型内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    jieba.initialize()
    main()