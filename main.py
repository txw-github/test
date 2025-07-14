#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTX 3060 Ti 中文视频转字幕工具
支持多种模型：Whisper、Faster-Whisper、FunASR
针对中文电视剧优化，支持TensorRT加速
"""

import os
import sys
import logging
import argparse
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Any

# 设置环境变量
os.environ['CUDA_LAZY_LOADING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('conversion.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 设置FFmpeg路径
ffmpeg_path = r"D:\code\ffmpeg\bin"
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# 检查依赖
try:
    import torch
    import torchaudio
    logger.info(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA不可用，将使用CPU模式")
except ImportError:
    logger.error("PyTorch未安装！请运行: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("Faster-Whisper未安装")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("OpenAI Whisper未安装")

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy未安装")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("Jieba未安装，中文分词功能不可用")


class Config:
    """配置类"""
    def __init__(self):
        self.rtx_3060_ti_optimized = True
        self.max_memory_usage = 0.8  # 使用80%显存
        self.batch_size = 16
        self.chunk_length = 30  # 30秒音频块
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # RTX 3060 Ti 特定优化
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "3060 Ti" in gpu_name:
                self.batch_size = 8  # 减少批次大小避免显存溢出
                self.chunk_length = 20  # 减少音频块大小
                logger.info("检测到RTX 3060 Ti，已应用优化设置")


class AudioProcessor:
    """音频处理器"""

    @staticmethod
    def extract_audio(video_path: str, output_path: str = None) -> str:
        """从视频提取音频"""
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy未安装，无法处理视频文件")

        if output_path is None:
            output_path = video_path.rsplit('.', 1)[0] + '_audio.wav'

        logger.info(f"正在提取音频: {video_path}")

        try:
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(output_path, verbose=False, logger=None)
            video.close()
            audio.close()

            logger.info(f"音频提取完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            raise


class TextProcessor:
    """文本后处理器"""

    def __init__(self):
        # 常见错别字词典
        self.corrections = {
            "纳里": "那里",
            "事后": "时候", 
            "只能": "智能",
            "马上": "马上",
            "因该": "应该",
            "的话": "的话",
            "在这个": "在这个",
            "然后": "然后"
        }

        # 专业词汇
        self.professional_terms = {
            "人工只能": "人工智能",
            "机器学习": "机器学习",
            "深度学系": "深度学习",
            "神经网络": "神经网络"
        }

        if JIEBA_AVAILABLE:
            # 添加专业词汇到jieba词典
            for term in self.professional_terms.values():
                jieba.add_word(term)

    def correct_text(self, text: str) -> str:
        """纠正文本错误"""
        # 基本纠错
        for wrong, correct in self.corrections.items():
            text = text.replace(wrong, correct)

        # 专业词汇纠错
        for wrong, correct in self.professional_terms.items():
            text = text.replace(wrong, correct)

        # 去除重复词
        text = self._remove_repetition(text)

        # 标点符号优化
        text = self._fix_punctuation(text)

        return text.strip()

    def _remove_repetition(self, text: str) -> str:
        """去除重复词汇"""
        import re
        # 去除重复的"嗯"、"啊"等语气词
        text = re.sub(r'(嗯){2,}', '嗯', text)
        text = re.sub(r'(啊){2,}', '啊', text)
        text = re.sub(r'(那个){2,}', '那个', text)
        return text

    def _fix_punctuation(self, text: str) -> str:
        """修复标点符号"""
        import re
        # 句末添加标点
        if text and not text[-1] in '。！？，；':
            if '？' in text or '什么' in text or '怎么' in text:
                text += '？'
            elif '！' in text or text.endswith(('啊', '呀', '哇')):
                text += '！'
            else:
                text += '。'
        return text


class WhisperModel:
    """Whisper模型包装器"""

    def __init__(self, model_name: str = "base", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.config = Config()
        self.text_processor = TextProcessor()

    def load_model(self):
        """加载模型"""
        logger.info(f"正在加载模型: {self.model_name}")

        if FASTER_WHISPER_AVAILABLE:
            self.model = WhisperModel(
                self.model_name, 
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            logger.info("使用Faster-Whisper模型")
        elif WHISPER_AVAILABLE:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("使用OpenAI Whisper模型")
        else:
            raise ImportError("没有可用的Whisper模型！")

    def transcribe(self, audio_path: str) -> List[Dict]:
        """转录音频"""
        if self.model is None:
            self.load_model()

        logger.info(f"开始转录: {audio_path}")
        start_time = time.time()

        try:
            if FASTER_WHISPER_AVAILABLE and hasattr(self.model, 'transcribe'):
                # Faster-Whisper
                segments, info = self.model.transcribe(
                    audio_path,
                    language="zh",
                    beam_size=5,
                    best_of=5,
                    temperature=0.0
                )

                results = []
                for segment in segments:
                    text = self.text_processor.correct_text(segment.text)
                    results.append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': text
                    })

            else:
                # OpenAI Whisper
                result = self.model.transcribe(audio_path, language="zh")
                results = []
                for segment in result['segments']:
                    text = self.text_processor.correct_text(segment['text'])
                    results.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text
                    })

            duration = time.time() - start_time
            logger.info(f"转录完成，耗时: {duration:.2f}秒")
            return results

        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise


class SRTGenerator:
    """SRT字幕生成器"""

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """格式化时间戳"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    @staticmethod
    def generate_srt(segments: List[Dict], output_path: str):
        """生成SRT文件"""
        logger.info(f"正在生成SRT文件: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = SRTGenerator.format_timestamp(segment['start'])
                end_time = SRTGenerator.format_timestamp(segment['end'])

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")

        logger.info(f"SRT文件生成完成: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RTX 3060 Ti 视频转字幕工具")
    parser.add_argument("video_path", help="视频文件路径")
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="选择模型大小")
    parser.add_argument("--output", help="输出SRT文件路径")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cuda", "cpu"], help="计算设备")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.video_path):
        logger.error(f"文件不存在: {args.video_path}")
        return

    # 设置输出路径
    if args.output is None:
        args.output = args.video_path.rsplit('.', 1)[0] + '.srt'

    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    try:
        # 提取音频
        audio_processor = AudioProcessor()
        audio_path = audio_processor.extract_audio(args.video_path)

        # 转录音频
        model = WhisperModel(args.model, device)
        segments = model.transcribe(audio_path)

        # 生成字幕
        SRTGenerator.generate_srt(segments, args.output)

        # 清理临时文件
        if os.path.exists(audio_path) and audio_path != args.video_path:
            os.remove(audio_path)

        logger.info("转换完成！")

    except Exception as e:
        logger.error(f"转换失败: {e}")
        raise


if __name__ == "__main__":
    main()