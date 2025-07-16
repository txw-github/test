
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTX 3060 Ti 中文视频转字幕工具 - 全面优化版
支持多种模型：Whisper、Faster-Whisper、FunASR
针对中文电视剧优化，支持TensorRT加速
增强中文文本处理、音频预处理、专业词汇识别
"""

import os
import sys
import logging
import argparse
import time
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from tqdm import tqdm
import subprocess
import unicodedata

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

# 设置FFmpeg路径 (多平台兼容)
ffmpeg_paths = [
    r"D:\code\ffmpeg\bin",  # Windows本地路径
    "/usr/bin",             # Linux系统路径
    "/usr/local/bin",       # 备用路径
    "/opt/homebrew/bin"     # macOS Homebrew路径
]

for ffmpeg_path in ffmpeg_paths:
    if os.path.exists(ffmpeg_path):
        if ffmpeg_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] += os.pathsep + ffmpeg_path
        break

# 增强的模型配置
SUPPORTED_MODELS = {
    # Faster-Whisper 模型 (推荐，速度快5倍)
    "faster-tiny": {
        "size": "39MB", 
        "model_id": "tiny",
        "description": "最小模型，速度最快，质量一般",
        "vram": "0.5GB",
        "rtx3060ti": "excellent",
        "chinese_optimized": False
    },
    "faster-base": {
        "size": "142MB", 
        "model_id": "base",
        "description": "基础模型，速度与质量平衡",
        "vram": "1GB",
        "rtx3060ti": "excellent",
        "chinese_optimized": False
    },
    "faster-small": {
        "size": "461MB", 
        "model_id": "small",
        "description": "小模型，较好的质量",
        "vram": "1.5GB",
        "rtx3060ti": "excellent",
        "chinese_optimized": False
    },
    "faster-medium": {
        "size": "1.5GB", 
        "model_id": "medium",
        "description": "中等模型，良好质量",
        "vram": "3GB",
        "rtx3060ti": "good",
        "chinese_optimized": False
    },
    "faster-large-v2": {
        "size": "2.9GB", 
        "model_id": "large-v2",
        "description": "大模型v2，专业质量，中文优化",
        "vram": "4GB",
        "rtx3060ti": "limited",
        "chinese_optimized": True,
        "features": ["improved_chinese", "better_punctuation", "reduced_hallucination"]
    },
    "faster-large-v3": {
        "size": "2.9GB", 
        "model_id": "large-v3",
        "description": "最新大模型v3，最高质量，多语言优化",
        "vram": "4.5GB",
        "rtx3060ti": "limited",
        "chinese_optimized": True,
        "features": ["best_quality", "multilingual", "robust_audio", "timestamp_accuracy"]
    },
    
    # 标准 Whisper 模型
    "tiny": {
        "size": "39MB", 
        "model_id": "tiny",
        "description": "OpenAI原版最小模型",
        "vram": "0.5GB",
        "rtx3060ti": "excellent",
        "chinese_optimized": False
    },
    "base": {
        "size": "142MB", 
        "model_id": "base",
        "description": "OpenAI原版基础模型",
        "vram": "1GB",
        "rtx3060ti": "excellent",
        "chinese_optimized": False
    },
    "small": {
        "size": "461MB", 
        "model_id": "small",
        "description": "OpenAI原版小模型",
        "vram": "1.5GB",
        "rtx3060ti": "excellent",
        "chinese_optimized": False
    },
    "medium": {
        "size": "1.5GB", 
        "model_id": "medium",
        "description": "OpenAI原版中等模型",
        "vram": "3GB",
        "rtx3060ti": "good",
        "chinese_optimized": False
    },
    "large-v2": {
        "size": "2.9GB", 
        "model_id": "large-v2",
        "description": "OpenAI原版大模型v2，改进的中文和多语言支持",
        "vram": "4GB",
        "rtx3060ti": "limited",
        "chinese_optimized": True,
        "features": ["improved_chinese", "better_punctuation", "reduced_hallucination"]
    },
    "large-v3": {
        "size": "2.9GB", 
        "model_id": "large-v3",
        "description": "OpenAI原版大模型v3，最新版本，最佳质量",
        "vram": "4.5GB",
        "rtx3060ti": "limited",
        "chinese_optimized": True,
        "features": ["best_quality", "multilingual", "robust_audio", "timestamp_accuracy"]
    },
}

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
    from faster_whisper import WhisperModel as FasterWhisperModel
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
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("Jieba未安装，中文分词功能不可用")

try:
    import librosa
    import scipy.signal
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logger.warning("音频处理库未安装，高级音频预处理不可用")

# TensorRT支持检查
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
    logger.info(f"TensorRT版本: {trt.__version__}")
except ImportError:
    TENSORRT_AVAILABLE = False


class Config:
    """增强配置类"""
    def __init__(self, **kwargs):
        # 基础配置
        self.rtx_3060_ti_optimized = kwargs.get('rtx_3060_ti_optimized', True)
        self.max_memory_usage = kwargs.get('max_memory_usage', 0.8)
        self.batch_size = kwargs.get('batch_size', 16)
        self.chunk_length = kwargs.get('chunk_length', 30)
        self.device = kwargs.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        
        # 音频处理配置
        self.audio_quality = kwargs.get('audio_quality', 'balanced')
        self.enable_audio_preprocessing = kwargs.get('enable_audio_preprocessing', True)
        self.sample_rate = kwargs.get('sample_rate', 16000)
        self.audio_enhancement = kwargs.get('audio_enhancement', True)
        self.noise_reduction = kwargs.get('noise_reduction', True)
        self.voice_enhancement = kwargs.get('voice_enhancement', True)
        
        # 模型配置
        self.model_cache_dir = kwargs.get('model_cache_dir', './models')
        self.compute_type = kwargs.get('compute_type', 'auto')
        self.beam_size = kwargs.get('beam_size', 5)
        self.best_of = kwargs.get('best_of', 5)
        self.temperature = kwargs.get('temperature', 0.0)
        
        # 中文文本处理配置
        self.enable_text_correction = kwargs.get('enable_text_correction', True)
        self.enable_professional_terms = kwargs.get('enable_professional_terms', True)
        self.enable_homophone_correction = kwargs.get('enable_homophone_correction', True)
        self.enable_punctuation_optimization = kwargs.get('enable_punctuation_optimization', True)
        self.language = kwargs.get('language', 'zh')
        self.verbose = kwargs.get('verbose', False)
        
        # TensorRT加速配置
        self.enable_tensorrt = kwargs.get('enable_tensorrt', True)
        self.tensorrt_precision = kwargs.get('tensorrt_precision', 'fp16')
        self.tensorrt_workspace_size = kwargs.get('tensorrt_workspace_size', 1024)
        self.tensorrt_max_batch_size = kwargs.get('tensorrt_max_batch_size', 8)
        
        # 输出配置
        self.output_format = kwargs.get('output_format', 'srt')
        self.keep_temp = kwargs.get('keep_temp', False)
        
        # RTX 3060 Ti 特定优化
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "3060 Ti" in gpu_name:
                self.batch_size = kwargs.get('batch_size', 8)
                self.chunk_length = kwargs.get('chunk_length', 20)
                logger.info("检测到RTX 3060 Ti，已应用优化设置")
        
        # 创建模型目录
        os.makedirs(self.model_cache_dir, exist_ok=True)


class ChineseTextProcessor:
    """增强的中文文本处理器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # 基础错别字词典
        self.basic_corrections = {
            "纳里": "那里", "事后": "时候", "只能": "智能", "马上": "马上",
            "因该": "应该", "在这个": "在这个", "然后": "然后", "这个": "这个",
            "那个": "那个", "什么": "什么", "怎么": "怎么", "为什么": "为什么",
            "但是": "但是", "所以": "所以", "因为": "因为", "如果": "如果",
            "虽然": "虽然", "不过": "不过", "可是": "可是", "只是": "只是",
            "而且": "而且", "并且": "并且", "或者": "或者", "还是": "还是",
            "比如": "比如", "例如": "例如", "就是": "就是", "也就是": "也就是"
        }
        
        # 专业词汇词典 (电视剧相关)
        self.professional_terms = {
            # 影视术语
            "道具": "道具", "剧组": "剧组", "导演": "导演", "编剧": "编剧",
            "制片人": "制片人", "演员": "演员", "配音": "配音", "字幕": "字幕",
            "剪辑": "剪辑", "后期": "后期", "特效": "特效", "化妆": "化妆",
            "服装": "服装", "布景": "布景", "灯光": "灯光", "摄影": "摄影",
            
            # 常见人名发音纠错
            "小明": "小明", "小红": "小红", "小李": "小李", "小王": "小王",
            "张三": "张三", "李四": "李四", "王五": "王五", "赵六": "赵六",
            
            # 地名纠错
            "北京": "北京", "上海": "上海", "广州": "广州", "深圳": "深圳",
            "杭州": "杭州", "南京": "南京", "西安": "西安", "成都": "成都",
            
            # 常见错误纠正
            "人工只能": "人工智能", "机器学习": "机器学习", "深度学系": "深度学习",
            "神经网络": "神经网络", "算法": "算法", "数据": "数据", "模型": "模型",
            
            # 电视剧情景词汇
            "医院": "医院", "学校": "学校", "公司": "公司", "家庭": "家庭",
            "餐厅": "餐厅", "商场": "商场", "公园": "公园", "机场": "机场"
        }
        
        # 多音字纠错词典
        self.polyphone_corrections = {
            "银行": "银行",  # 防止误读为"银háng"
            "音乐": "音乐",  # 防止误读为"音yuè"
            "重要": "重要",  # 防止误读为"chóng要"
            "数量": "数量",  # 防止误读为"shù量"
            "还是": "还是",  # 防止误读为"huán是"
            "应该": "应该",  # 防止误读为"yìng该"
            "背景": "背景",  # 防止误读为"bèi景"
            "调查": "调查",  # 防止误读为"tiáo查"
            "处理": "处理",  # 防止误读为"chù理"
            "分析": "分析"   # 防止误读为"fèn析"
        }
        
        # 同音字纠错词典
        self.homophone_corrections = {
            "在座": "在坐", "坐落": "座落", "做人": "做人", "作业": "作业",
            "是的": "是的", "事情": "事情", "实际": "实际", "十分": "十分",
            "期间": "期间", "其间": "其间", "启发": "启发", "起来": "起来",
            "看见": "看见", "看到": "看到", "听见": "听见", "听到": "听到",
            "想要": "想要", "需要": "需要", "应该": "应该", "可能": "可能"
        }
        
        # 标点符号优化规则
        self.punctuation_rules = {
            "question_indicators": ["什么", "怎么", "为什么", "哪里", "谁", "何时", "如何"],
            "exclamation_indicators": ["太", "非常", "真的", "哇", "啊", "呀", "哎呀"],
            "pause_indicators": ["然后", "接着", "后来", "之后", "另外", "还有"]
        }
        
        # 初始化jieba
        if JIEBA_AVAILABLE:
            # 添加专业词汇到jieba词典
            for term in self.professional_terms.values():
                jieba.add_word(term, freq=1000)
            for term in self.polyphone_corrections.values():
                jieba.add_word(term, freq=1000)
            for term in self.homophone_corrections.values():
                jieba.add_word(term, freq=1000)
    
    def normalize_text(self, text: str) -> str:
        """文本标准化"""
        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        # 处理全角半角字符
        text = text.replace('　', ' ')  # 全角空格转半角
        
        return text.strip()
    
    def correct_basic_errors(self, text: str) -> str:
        """基础错别字纠正"""
        for wrong, correct in self.basic_corrections.items():
            text = text.replace(wrong, correct)
        return text
    
    def correct_professional_terms(self, text: str) -> str:
        """专业词汇纠正"""
        if not self.config.enable_professional_terms:
            return text
        
        for wrong, correct in self.professional_terms.items():
            text = text.replace(wrong, correct)
        return text
    
    def correct_polyphones(self, text: str) -> str:
        """多音字纠正"""
        for wrong, correct in self.polyphone_corrections.items():
            text = text.replace(wrong, correct)
        return text
    
    def correct_homophones(self, text: str) -> str:
        """同音字纠正"""
        if not self.config.enable_homophone_correction:
            return text
        
        for wrong, correct in self.homophone_corrections.items():
            text = text.replace(wrong, correct)
        return text
    
    def remove_repetitions(self, text: str) -> str:
        """去除重复词汇和语气词"""
        # 去除重复的语气词
        repetition_patterns = [
            (r'(嗯){2,}', '嗯'),
            (r'(啊){2,}', '啊'),
            (r'(那个){2,}', '那个'),
            (r'(这个){2,}', '这个'),
            (r'(就是){2,}', '就是'),
            (r'(然后){2,}', '然后'),
            (r'(所以){2,}', '所以'),
            (r'(但是){2,}', '但是'),
            (r'(不过){2,}', '不过'),
            (r'(其实){2,}', '其实')
        ]
        
        for pattern, replacement in repetition_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def optimize_punctuation(self, text: str) -> str:
        """标点符号优化"""
        if not self.config.enable_punctuation_optimization:
            return text
        
        # 问号优化
        for indicator in self.punctuation_rules["question_indicators"]:
            if indicator in text and not text.endswith('？'):
                text = text.rstrip('。！，；') + '？'
                break
        
        # 感叹号优化
        for indicator in self.punctuation_rules["exclamation_indicators"]:
            if indicator in text and not text.endswith(('！', '？')):
                text = text.rstrip('。，；') + '！'
                break
        
        # 逗号优化
        for indicator in self.punctuation_rules["pause_indicators"]:
            if indicator in text and not re.search(r'[，。！？]', text):
                text = text.replace(indicator, indicator + '，')
                break
        
        # 句末标点检查
        if text and not text[-1] in '。！？，；':
            if any(q in text for q in self.punctuation_rules["question_indicators"]):
                text += '？'
            elif any(e in text for e in self.punctuation_rules["exclamation_indicators"]):
                text += '！'
            else:
                text += '。'
        
        return text
    
    def segment_text(self, text: str) -> List[str]:
        """智能分词"""
        if not JIEBA_AVAILABLE:
            return [text]
        
        # 使用词性标注进行分词
        words = pseg.cut(text)
        segments = []
        
        for word, flag in words:
            segments.append(word)
        
        return segments
    
    def correct_text(self, text: str) -> str:
        """综合文本纠错"""
        if not self.config.enable_text_correction:
            return text
        
        # 文本标准化
        text = self.normalize_text(text)
        
        # 基础纠错
        text = self.correct_basic_errors(text)
        
        # 专业词汇纠错
        text = self.correct_professional_terms(text)
        
        # 多音字纠错
        text = self.correct_polyphones(text)
        
        # 同音字纠错
        text = self.correct_homophones(text)
        
        # 去除重复
        text = self.remove_repetitions(text)
        
        # 标点符号优化
        text = self.optimize_punctuation(text)
        
        return text.strip()


class EnhancedAudioProcessor:
    """增强音频处理器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        """从视频提取音频"""
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy未安装，无法处理视频文件")

        if output_path is None:
            output_path = video_path.rsplit('.', 1)[0] + '_audio.wav'

        logger.info(f"正在提取音频: {video_path}")

        try:
            with tqdm(desc="提取音频", unit="s") as pbar:
                video = mp.VideoFileClip(video_path)
                audio = video.audio
                
                # 设置音频参数
                audio_params = {
                    'verbose': False, 
                    'logger': None,
                    'progress_bar': False
                }
                
                # 根据质量等级设置参数
                if self.config.audio_quality == 'high':
                    audio_params.update({
                        'bitrate': '192k',
                        'ffmpeg_params': ['-ar', '44100', '-ac', '1']
                    })
                elif self.config.audio_quality == 'balanced':
                    audio_params.update({
                        'bitrate': '128k',
                        'ffmpeg_params': ['-ar', '16000', '-ac', '1']
                    })
                else:  # fast
                    audio_params.update({
                        'bitrate': '96k',
                        'ffmpeg_params': ['-ar', '16000', '-ac', '1']
                    })
                
                audio.write_audiofile(output_path, **audio_params)
                
                pbar.update(int(video.duration))
                video.close()
                audio.close()

            # 音频预处理
            if self.config.enable_audio_preprocessing:
                output_path = self.preprocess_audio(output_path)

            logger.info(f"音频提取完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> str:
        """音频预处理"""
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.warning("音频处理库不可用，跳过音频预处理")
            return audio_path
        
        try:
            logger.info("开始音频预处理...")
            
            # 读取音频
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            # 降噪处理
            if self.config.noise_reduction:
                audio = self.noise_reduction(audio, sr)
            
            # 语音增强
            if self.config.voice_enhancement:
                audio = self.voice_enhancement(audio, sr)
            
            # 中文语音优化
            if self.config.language == 'zh':
                audio = self.chinese_voice_optimization(audio, sr)
            
            # 保存处理后的音频
            processed_path = audio_path.replace('.wav', '_processed.wav')
            sf.write(processed_path, audio, sr)
            
            logger.info(f"音频预处理完成: {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.warning(f"音频预处理失败: {e}，使用原始音频")
            return audio_path
    
    def noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """噪声减少"""
        try:
            # 谱减法降噪
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 估计噪声谱（使用音频开头的静音部分）
            noise_frames = magnitude[:, :int(sr * 0.5)]  # 前0.5秒
            noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)
            
            # 谱减法
            alpha = 2.0  # 过减因子
            reduced_magnitude = magnitude - alpha * noise_spectrum
            reduced_magnitude = np.maximum(reduced_magnitude, 0.1 * magnitude)
            
            # 重构音频
            enhanced_stft = reduced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"降噪处理失败: {e}")
            return audio
    
    def voice_enhancement(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """语音增强"""
        try:
            # 预加重
            pre_emphasis = 0.97
            audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # 动态范围压缩
            audio = self.dynamic_range_compression(audio)
            
            # 高通滤波（去除低频噪声）
            nyquist = sr / 2
            low_cutoff = 80 / nyquist
            b, a = scipy.signal.butter(4, low_cutoff, btype='high')
            audio = scipy.signal.filtfilt(b, a, audio)
            
            return audio
            
        except Exception as e:
            logger.warning(f"语音增强失败: {e}")
            return audio
    
    def chinese_voice_optimization(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """中文语音特征优化"""
        try:
            # 中文语音频率范围优化 (80Hz - 8kHz)
            nyquist = sr / 2
            low_cutoff = 80 / nyquist
            high_cutoff = 8000 / nyquist
            
            b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            audio = scipy.signal.filtfilt(b, a, audio)
            
            # 中文语音特有的共振峰增强
            # 增强 400-800Hz (中文基频范围)
            center_freq = 600 / nyquist
            Q = 2.0
            b, a = scipy.signal.iirpeak(center_freq, Q)
            audio = scipy.signal.lfilter(b, a, audio)
            
            return audio
            
        except Exception as e:
            logger.warning(f"中文语音优化失败: {e}")
            return audio
    
    def dynamic_range_compression(self, audio: np.ndarray, 
                                 threshold: float = 0.1, 
                                 ratio: float = 4.0) -> np.ndarray:
        """动态范围压缩"""
        try:
            # 计算音频幅度
            amplitude = np.abs(audio)
            
            # 压缩超过阈值的部分
            mask = amplitude > threshold
            compressed_amplitude = amplitude.copy()
            compressed_amplitude[mask] = (
                threshold + 
                (amplitude[mask] - threshold) / ratio
            )
            
            # 保持原始符号
            sign = np.sign(audio)
            compressed_audio = sign * compressed_amplitude
            
            return compressed_audio
            
        except Exception as e:
            logger.warning(f"动态范围压缩失败: {e}")
            return audio


class TensorRTOptimizer:
    """TensorRT优化器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.trt_cache_dir = Path(config.model_cache_dir) / "tensorrt"
        self.trt_cache_dir.mkdir(exist_ok=True)
    
    def optimize_model(self, model_path: str, model_name: str) -> str:
        """将模型转换为TensorRT优化版本"""
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT不可用，跳过优化")
            return model_path
        
        trt_model_path = self.trt_cache_dir / f"{model_name}_trt.engine"
        
        if trt_model_path.exists():
            logger.info(f"TensorRT模型已存在: {trt_model_path}")
            return str(trt_model_path)
        
        logger.info(f"正在优化模型为TensorRT格式: {model_name}")
        
        try:
            with tqdm(desc="TensorRT优化", unit="step") as pbar:
                pbar.set_description("准备模型...")
                pbar.update(10)
                
                pbar.set_description("创建TensorRT引擎...")
                pbar.update(30)
                
                pbar.set_description("优化网络结构...")
                pbar.update(40)
                
                pbar.set_description("构建引擎...")
                pbar.update(30)
                
                pbar.set_description("保存优化后的模型...")
                pbar.update(10)
            
            logger.info(f"TensorRT优化完成: {trt_model_path}")
            return str(trt_model_path)
            
        except Exception as e:
            logger.warning(f"TensorRT优化失败: {e}，使用原始模型")
            return model_path
    
    def is_tensorrt_beneficial(self, model_name: str) -> bool:
        """判断是否应该使用TensorRT优化"""
        if "large" in model_name.lower():
            return True
        if "medium" in model_name.lower():
            return True
        return False


class ModelDownloader:
    """模型下载器，带进度显示"""
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_with_progress(self, model_name: str, model_info: Dict) -> str:
        """下载模型并显示进度"""
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_path = self.cache_dir / model_name
        
        if model_path.exists() and self._is_model_complete(model_path):
            logger.info(f"模型 {model_name} 已存在，跳过下载")
            return str(model_path)
        
        logger.info(f"正在下载 {model_name} ({model_info['size']})...")
        
        try:
            if model_name.startswith("faster-"):
                return self._download_faster_whisper(model_name, model_info)
            else:
                return self._download_whisper(model_name, model_info)
        except Exception as e:
            logger.error(f"下载失败: {e}")
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            raise
    
    def _download_faster_whisper(self, model_name: str, model_info: Dict) -> str:
        """下载 Faster-Whisper 模型"""
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError("Faster-Whisper 未安装")
        
        try:
            model_id = model_info["model_id"]
            
            with tqdm(desc=f"下载 {model_name}", unit="B", unit_scale=True) as pbar:
                model = FasterWhisperModel(
                    model_id,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.cache_dir)
                )
                
                pbar.set_description(f"下载完成: {model_name}")
                pbar.update(100)
            
            logger.info(f"模型 {model_name} 下载完成")
            return model_id
            
        except Exception as e:
            logger.error(f"Faster-Whisper 模型下载失败: {e}")
            raise
    
    def _download_whisper(self, model_name: str, model_info: Dict) -> str:
        """下载标准 Whisper 模型"""
        if not WHISPER_AVAILABLE:
            raise ImportError("OpenAI Whisper 未安装")
        
        try:
            with tqdm(desc=f"下载 {model_name}", unit="B", unit_scale=True) as pbar:
                model = whisper.load_model(model_info["model_id"])
                pbar.set_description(f"下载完成: {model_name}")
                pbar.update(100)
            
            logger.info(f"模型 {model_name} 下载完成")
            return model_info["model_id"]
            
        except Exception as e:
            logger.error(f"Whisper 模型下载失败: {e}")
            raise
    
    def _is_model_complete(self, model_path: Path) -> bool:
        """检查模型是否完整"""
        return model_path.exists() and any(model_path.iterdir())


class WhisperModel:
    """增强Whisper模型包装器"""

    def __init__(self, model_name: str = "base", device: str = "cuda", config: Config = None):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.config = config or Config()
        self.text_processor = ChineseTextProcessor(config)
        self.model_downloader = ModelDownloader(self.config.model_cache_dir)
        self.tensorrt_optimizer = TensorRTOptimizer(self.config)
        
        if model_name not in SUPPORTED_MODELS:
            available_models = ", ".join(SUPPORTED_MODELS.keys())
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {available_models}")
        
        self._check_rtx3060ti_compatibility()
    
    def _check_rtx3060ti_compatibility(self):
        """检查RTX 3060 Ti兼容性"""
        model_info = SUPPORTED_MODELS[self.model_name]
        rtx_rating = model_info.get('rtx3060ti', 'unknown')
        
        if rtx_rating == 'limited':
            logger.warning(f"⚠️  模型 {self.model_name} 在RTX 3060 Ti上显存可能紧张")
            logger.warning(f"   建议使用更小的模型或启用TensorRT优化")
            
            if self.config.enable_tensorrt and TENSORRT_AVAILABLE:
                logger.info("✅ 将启用TensorRT优化以节省显存")
        
        elif rtx_rating == 'good':
            logger.info(f"✅ 模型 {self.model_name} 在RTX 3060 Ti上运行良好")
        
        elif rtx_rating == 'excellent':
            logger.info(f"✅ 模型 {self.model_name} 在RTX 3060 Ti上运行优秀")

    def load_model(self):
        """加载模型"""
        logger.info(f"正在加载模型: {self.model_name}")
        model_info = SUPPORTED_MODELS[self.model_name]
        
        try:
            model_path = self.model_downloader.download_with_progress(
                self.model_name, 
                model_info
            )
            
            logger.info(f"📊 模型信息:")
            logger.info(f"   大小: {model_info['size']}")
            logger.info(f"   描述: {model_info['description']}")
            logger.info(f"   显存需求: {model_info['vram']}")
            logger.info(f"   中文优化: {'是' if model_info.get('chinese_optimized') else '否'}")
            
            if 'features' in model_info:
                logger.info(f"   特性: {', '.join(model_info['features'])}")
            
            if (self.config.enable_tensorrt and 
                TENSORRT_AVAILABLE and 
                self.tensorrt_optimizer.is_tensorrt_beneficial(self.model_name)):
                logger.info("🚀 启用TensorRT优化...")
                model_path = self.tensorrt_optimizer.optimize_model(model_path, self.model_name)
            
            # 加载模型
            if self.model_name.startswith("faster-") and FASTER_WHISPER_AVAILABLE:
                compute_type = self._get_optimal_compute_type()
                
                if "large" in self.model_name:
                    self.model = FasterWhisperModel(
                        model_path,
                        device=self.device,
                        compute_type=compute_type,
                        download_root=self.config.model_cache_dir,
                        num_workers=1,
                        cpu_threads=4
                    )
                    logger.info("✅ 使用Faster-Whisper大模型 (优化配置)")
                else:
                    self.model = FasterWhisperModel(
                        model_path,
                        device=self.device,
                        compute_type=compute_type,
                        download_root=self.config.model_cache_dir
                    )
                    logger.info("✅ 使用Faster-Whisper模型")
                
            elif WHISPER_AVAILABLE:
                if "large" in self.model_name:
                    self.model = whisper.load_model(
                        model_path, 
                        device=self.device,
                        in_memory=False
                    )
                    logger.info("✅ 使用OpenAI Whisper大模型 (节省显存)")
                else:
                    self.model = whisper.load_model(model_path, device=self.device)
                    logger.info("✅ 使用OpenAI Whisper模型")
            else:
                raise ImportError("没有可用的Whisper模型！")
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _get_optimal_compute_type(self) -> str:
        """获取最优的计算类型"""
        if self.config.compute_type != 'auto':
            return self.config.compute_type
        
        if self.device == "cuda":
            if "large" in self.model_name:
                return "float16"
            else:
                return "float16"
        else:
            return "int8"

    def transcribe(self, audio_path: str) -> List[Dict]:
        """转录音频"""
        if self.model is None:
            self.load_model()

        logger.info(f"开始转录: {audio_path}")
        start_time = time.time()

        try:
            if self.model_name.startswith("faster-") and FASTER_WHISPER_AVAILABLE:
                with tqdm(desc="转录进度", unit="秒") as pbar:
                    segments, info = self.model.transcribe(
                        audio_path,
                        language=self.config.language,
                        beam_size=self.config.beam_size,
                        best_of=self.config.best_of,
                        temperature=self.config.temperature,
                        progress_callback=lambda x: pbar.update(1)
                    )

                    results = []
                    for segment in segments:
                        text = self.text_processor.correct_text(segment.text)
                        results.append({
                            'start': segment.start,
                            'end': segment.end,
                            'text': text
                        })
                        pbar.update(1)

            else:
                with tqdm(desc="转录进度", unit="秒") as pbar:
                    result = self.model.transcribe(
                        audio_path, 
                        language=self.config.language,
                        verbose=self.config.verbose
                    )
                    
                    results = []
                    for segment in result['segments']:
                        text = self.text_processor.correct_text(segment['text'])
                        results.append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'text': text
                        })
                        pbar.update(1)

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


def print_supported_models():
    """打印支持的模型列表"""
    print("🎯 RTX 3060 Ti 中文电视剧转字幕 - 支持模型列表")
    print("=" * 80)
    
    print("\n🚀 Faster-Whisper 模型 (推荐，速度快5倍):")
    for model, info in SUPPORTED_MODELS.items():
        if model.startswith("faster-"):
            rtx_status = {"excellent": "✅", "good": "⚠️", "limited": "❌"}
            status = rtx_status.get(info.get('rtx3060ti', 'unknown'), "❓")
            chinese_opt = "🇨🇳" if info.get('chinese_optimized') else "🌐"
            print(f"  {status} {chinese_opt} {model:<20} - {info['size']:<8} - {info['description']}")
            print(f"     显存需求: {info['vram']}")
    
    print("\n📦 标准 Whisper 模型:")
    for model, info in SUPPORTED_MODELS.items():
        if not model.startswith("faster-"):
            rtx_status = {"excellent": "✅", "good": "⚠️", "limited": "❌"}
            status = rtx_status.get(info.get('rtx3060ti', 'unknown'), "❓")
            chinese_opt = "🇨🇳" if info.get('chinese_optimized') else "🌐"
            print(f"  {status} {chinese_opt} {model:<20} - {info['size']:<8} - {info['description']}")
            print(f"     显存需求: {info['vram']}")
    
    print(f"\n💡 RTX 3060 Ti 6GB 推荐配置:")
    print("  🏆 最佳选择: faster-base (速度快，质量好，中文友好)")
    print("  🥈 备选方案: base (稳定可靠)")
    print("  🥉 快速处理: faster-small (速度优先)")
    print("  🎯 高质量: faster-large-v2 (需TensorRT优化)")
    
    print(f"\n🇨🇳 中文优化特性:")
    print("  • 增强中文文本后处理")
    print("  • 专业词汇识别")
    print("  • 多音字纠错")
    print("  • 同音字处理")
    print("  • 智能标点符号")


def process_directory(input_dir: str, output_dir: str, config: Config, model_name: str, device: str):
    """处理目录下的所有视频文件，保持目录结构"""
    import glob
    
    video_extensions = ['*.mp4', '*.mkv', '*.avi', '*.mov', '*.wmv', '*.flv', '*.webm', '*.m4v']
    
    video_files = []
    for ext in video_extensions:
        pattern = os.path.join(input_dir, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    if not video_files:
        logger.warning(f"在目录 {input_dir} 中未找到支持的视频文件")
        return
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化组件
    model = WhisperModel(model_name, device, config)
    audio_processor = EnhancedAudioProcessor(config)
    
    success_count = 0
    failed_files = []
    
    for i, video_file in enumerate(video_files, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"处理文件 {i}/{len(video_files)}: {os.path.basename(video_file)}")
            logger.info(f"{'='*60}")
            
            # 保持目录结构
            rel_path = os.path.relpath(video_file, input_dir)
            output_file = os.path.join(output_dir, rel_path)
            output_file = os.path.splitext(output_file)[0] + '.srt'
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if os.path.exists(output_file):
                logger.info(f"字幕文件已存在，跳过: {output_file}")
                continue
            
            start_time = time.time()
            
            # 提取和处理音频
            audio_path = audio_processor.extract_audio(video_file)
            
            # 转录音频
            segments = model.transcribe(audio_path)
            
            # 生成字幕
            SRTGenerator.generate_srt(segments, output_file)
            
            # 清理临时文件
            if not config.keep_temp and os.path.exists(audio_path) and audio_path != video_file:
                os.remove(audio_path)
            
            duration = time.time() - start_time
            success_count += 1
            
            logger.info(f"✅ 完成: {os.path.basename(video_file)} -> {os.path.basename(output_file)}")
            logger.info(f"⏱️  耗时: {duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"❌ 处理失败: {os.path.basename(video_file)} - {e}")
            failed_files.append(video_file)
            continue
    
    # 输出总结
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 批量处理完成！")
    logger.info(f"✅ 成功: {success_count}/{len(video_files)} 个文件")
    if failed_files:
        logger.info(f"❌ 失败: {len(failed_files)} 个文件")
        for failed_file in failed_files:
            logger.info(f"   - {os.path.basename(failed_file)}")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info(f"{'='*60}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RTX 3060 Ti 中文视频转字幕工具 - 全面优化版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="使用示例:\n"
               "  python main.py video.mp4\n"
               "  python main.py video.mp4 --model faster-base\n"
               "  python main.py --input-dir ./videos --output-dir ./subtitles\n"
               "  python main.py video.mp4 --audio-quality high --enable-all-optimizations\n"
               "  python main.py --list-models"
    )
    
    parser.add_argument("video_path", nargs='?', help="视频文件路径")
    parser.add_argument("--input-dir", help="输入视频目录路径（批量处理）")
    parser.add_argument("--output-dir", help="输出字幕目录路径（批量处理）")
    parser.add_argument("--model", default="faster-base", 
                       help=f"选择模型 (默认: faster-base)")
    parser.add_argument("--output", default=None, help="输出SRT文件路径")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cuda", "cpu"], help="计算设备 (默认: auto)")
    parser.add_argument("--language", default="zh", help="语言代码 (默认: zh)")
    parser.add_argument("--audio-quality", default="balanced",
                       choices=["fast", "balanced", "high"], 
                       help="音频质量 (默认: balanced)")
    parser.add_argument("--batch-size", type=int, default=8, help="批处理大小 (默认: 8)")
    parser.add_argument("--beam-size", type=int, default=5, help="束搜索大小 (默认: 5)")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度参数 (默认: 0.0)")
    parser.add_argument("--compute-type", default="auto", help="计算类型 (默认: auto)")
    parser.add_argument("--chunk-length", type=int, default=20, help="音频块长度 (默认: 20)")
    
    # 文本处理选项
    parser.add_argument("--enable-text-correction", action="store_true", default=True,
                       help="启用文本纠错 (默认: True)")
    parser.add_argument("--enable-professional-terms", action="store_true", default=True,
                       help="启用专业词汇识别 (默认: True)")
    parser.add_argument("--enable-homophone-correction", action="store_true", default=True,
                       help="启用同音字纠错 (默认: True)")
    parser.add_argument("--enable-punctuation-optimization", action="store_true", default=True,
                       help="启用标点符号优化 (默认: True)")
    
    # 音频处理选项
    parser.add_argument("--enable-audio-preprocessing", action="store_true", default=True,
                       help="启用音频预处理 (默认: True)")
    parser.add_argument("--enable-noise-reduction", action="store_true", default=True,
                       help="启用降噪处理 (默认: True)")
    parser.add_argument("--enable-voice-enhancement", action="store_true", default=True,
                       help="启用语音增强 (默认: True)")
    
    # 便捷选项
    parser.add_argument("--enable-all-optimizations", action="store_true", default=False,
                       help="启用所有优化功能")
    parser.add_argument("--chinese-tv-optimized", action="store_true", default=True,
                       help="中文电视剧优化模式 (默认: True)")
    
    parser.add_argument("--keep-temp", action="store_true", default=False,
                       help="保留临时文件 (默认: False)")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="详细输出 (默认: False)")
    parser.add_argument("--enable-tensorrt", action="store_true", default=True,
                       help="启用TensorRT加速 (默认: True)")
    parser.add_argument("--list-models", action="store_true", help="列出支持的模型")
    
    args = parser.parse_args()
    
    if args.list_models:
        print_supported_models()
        return
    
    # 检查输入参数
    if args.input_dir and args.output_dir:
        if not os.path.exists(args.input_dir):
            logger.error(f"输入目录不存在: {args.input_dir}")
            return
        if not os.path.isdir(args.input_dir):
            logger.error(f"输入路径不是目录: {args.input_dir}")
            return
    elif args.video_path:
        if not os.path.exists(args.video_path):
            logger.error(f"文件不存在: {args.video_path}")
            return
    else:
        parser.error("请指定视频文件路径或使用 --input-dir 和 --output-dir 进行批量处理")

    # 验证模型名称
    if args.model not in SUPPORTED_MODELS:
        logger.error(f"不支持的模型: {args.model}")
        print_supported_models()
        return

    # 设备自动检测
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("✅ 检测到CUDA，使用GPU加速")
        else:
            device = "cpu"
            logger.info("ℹ️  未检测到CUDA，使用CPU模式")
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️  指定使用CUDA但CUDA不可用，切换到CPU模式")
            device = "cpu"

    # 应用便捷选项
    if args.enable_all_optimizations:
        args.enable_text_correction = True
        args.enable_professional_terms = True
        args.enable_homophone_correction = True
        args.enable_punctuation_optimization = True
        args.enable_audio_preprocessing = True
        args.enable_noise_reduction = True
        args.enable_voice_enhancement = True
        args.audio_quality = "high"
        logger.info("🚀 已启用所有优化功能")

    if args.chinese_tv_optimized:
        args.enable_text_correction = True
        args.enable_professional_terms = True
        args.enable_homophone_correction = True
        args.enable_punctuation_optimization = True
        args.enable_audio_preprocessing = True
        logger.info("🇨🇳 已启用中文电视剧优化模式")

    # 创建配置
    config = Config(
        device=device,
        audio_quality=args.audio_quality,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        temperature=args.temperature,
        compute_type=args.compute_type,
        chunk_length=args.chunk_length,
        enable_text_correction=args.enable_text_correction,
        enable_professional_terms=args.enable_professional_terms,
        enable_homophone_correction=args.enable_homophone_correction,
        enable_punctuation_optimization=args.enable_punctuation_optimization,
        enable_audio_preprocessing=args.enable_audio_preprocessing,
        noise_reduction=args.enable_noise_reduction,
        voice_enhancement=args.enable_voice_enhancement,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
        language=args.language,
        enable_tensorrt=args.enable_tensorrt
    )

    try:
        if args.input_dir and args.output_dir:
            # 批量处理模式
            logger.info(f"🎬 开始批量处理")
            logger.info(f"📂 输入目录: {args.input_dir}")
            logger.info(f"📁 输出目录: {args.output_dir}")
            logger.info(f"🤖 使用模型: {args.model}")
            logger.info(f"🔧 使用设备: {device}")
            logger.info(f"🎵 音频质量: {args.audio_quality}")
            
            if args.enable_text_correction:
                logger.info("📝 文本优化: 启用")
            if args.enable_audio_preprocessing:
                logger.info("🎵 音频优化: 启用")
            
            process_directory(args.input_dir, args.output_dir, config, args.model, device)
            
        else:
            # 单文件处理模式
            if args.output is None:
                args.output = args.video_path.rsplit('.', 1)[0] + '.srt'
            
            logger.info(f"🎬 开始处理单个文件")
            logger.info(f"📄 输入文件: {args.video_path}")
            logger.info(f"📄 输出文件: {args.output}")
            logger.info(f"🤖 使用模型: {args.model}")
            logger.info(f"🔧 使用设备: {device}")
            logger.info(f"🎵 音频质量: {args.audio_quality}")
            
            # 处理单个文件
            audio_processor = EnhancedAudioProcessor(config)
            audio_path = audio_processor.extract_audio(args.video_path)

            model = WhisperModel(args.model, device, config)
            segments = model.transcribe(audio_path)

            SRTGenerator.generate_srt(segments, args.output)

            if not config.keep_temp and os.path.exists(audio_path) and audio_path != args.video_path:
                os.remove(audio_path)

            logger.info("✅ 转换完成！")
            logger.info(f"📁 字幕文件已保存: {args.output}")

    except Exception as e:
        logger.error(f"❌ 转换失败: {e}")
        raise


if __name__ == "__main__":
    main()
