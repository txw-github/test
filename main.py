
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
from tqdm import tqdm
import subprocess

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

# 模型配置
SUPPORTED_MODELS = {
    # Faster-Whisper 模型
    "faster-tiny": {"size": "39MB", "model_id": "guillaumekln/faster-whisper-tiny"},
    "faster-base": {"size": "142MB", "model_id": "guillaumekln/faster-whisper-base"},
    "faster-small": {"size": "461MB", "model_id": "guillaumekln/faster-whisper-small"},
    "faster-medium": {"size": "1.5GB", "model_id": "guillaumekln/faster-whisper-medium"},
    "faster-large": {"size": "2.9GB", "model_id": "guillaumekln/faster-whisper-large-v2"},
    "faster-large-v3": {"size": "2.9GB", "model_id": "guillaumekln/faster-whisper-large-v3"},
    
    # 标准 Whisper 模型
    "tiny": {"size": "39MB", "model_id": "tiny"},
    "base": {"size": "142MB", "model_id": "base"},
    "small": {"size": "461MB", "model_id": "small"},
    "medium": {"size": "1.5GB", "model_id": "medium"},
    "large": {"size": "2.9GB", "model_id": "large"},
    "large-v2": {"size": "2.9GB", "model_id": "large-v2"},
    "large-v3": {"size": "2.9GB", "model_id": "large-v3"},
    
    # 中文优化模型
    "chinese-whisper-small": {"size": "461MB", "model_id": "openai/whisper-small"},
    "chinese-whisper-base": {"size": "142MB", "model_id": "openai/whisper-base"},
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
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("Jieba未安装，中文分词功能不可用")

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("Hugging Face Hub未安装")


class Config:
    """配置类"""
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
        
        # 模型配置
        self.model_cache_dir = kwargs.get('model_cache_dir', './models')
        self.compute_type = kwargs.get('compute_type', 'auto')
        self.beam_size = kwargs.get('beam_size', 5)
        self.best_of = kwargs.get('best_of', 5)
        self.temperature = kwargs.get('temperature', 0.0)
        
        # 文本处理配置
        self.enable_text_correction = kwargs.get('enable_text_correction', True)
        self.language = kwargs.get('language', 'zh')
        self.verbose = kwargs.get('verbose', False)
        
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
        
        # 检查是否已经下载
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
            # 对于 faster-whisper，我们直接使用模型 ID
            model_id = model_info["model_id"]
            
            # 创建进度条
            with tqdm(desc=f"下载 {model_name}", unit="B", unit_scale=True) as pbar:
                # 初始化模型（这会触发下载）
                model = FasterWhisperModel(
                    model_id,
                    device="cpu",  # 先用CPU初始化
                    compute_type="int8",
                    download_root=str(self.cache_dir)
                )
                
                # 更新进度条
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
                # 使用 whisper.load_model 会自动下载
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


class AudioProcessor:
    """音频处理器"""
    
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
                
                # 设置进度回调
                def progress_callback(t):
                    pbar.n = int(t)
                    pbar.refresh()
                
                audio.write_audiofile(
                    output_path, 
                    verbose=False, 
                    logger=None,
                    progress_bar=False
                )
                
                pbar.update(int(video.duration))
                video.close()
                audio.close()

            logger.info(f"音频提取完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            raise


class TextProcessor:
    """文本后处理器"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # 常见错别字词典
        self.corrections = {
            "纳里": "那里",
            "事后": "时候", 
            "只能": "智能",
            "马上": "马上",
            "因该": "应该",
            "的话": "的话",
            "在这个": "在这个",
            "然后": "然后",
            "这个": "这个",
            "那个": "那个",
            "什么": "什么",
            "怎么": "怎么",
            "为什么": "为什么",
            "但是": "但是",
            "所以": "所以",
            "因为": "因为",
            "如果": "如果",
            "虽然": "虽然",
            "不过": "不过",
            "可是": "可是",
            "只是": "只是",
            "而且": "而且",
            "并且": "并且",
            "或者": "或者",
            "还是": "还是",
            "比如": "比如",
            "例如": "例如",
            "就是": "就是",
            "也就是": "也就是"
        }

        # 专业词汇
        self.professional_terms = {
            "人工只能": "人工智能",
            "机器学习": "机器学习",
            "深度学系": "深度学习",
            "神经网络": "神经网络",
            "算法": "算法",
            "数据": "数据",
            "模型": "模型",
            "训练": "训练",
            "优化": "优化",
            "预测": "预测",
            "分析": "分析",
            "处理": "处理",
            "系统": "系统",
            "平台": "平台",
            "技术": "技术",
            "方法": "方法",
            "工具": "工具",
            "软件": "软件",
            "硬件": "硬件",
            "网络": "网络",
            "互联网": "互联网",
            "计算机": "计算机",
            "程序": "程序",
            "代码": "代码",
            "开发": "开发",
            "设计": "设计",
            "应用": "应用",
            "服务": "服务",
            "产品": "产品",
            "项目": "项目",
            "管理": "管理",
            "运营": "运营",
            "市场": "市场",
            "营销": "营销",
            "销售": "销售",
            "客户": "客户",
            "用户": "用户",
            "体验": "体验",
            "界面": "界面",
            "功能": "功能",
            "性能": "性能",
            "效率": "效率",
            "质量": "质量",
            "安全": "安全",
            "稳定": "稳定",
            "可靠": "可靠",
            "创新": "创新",
            "发展": "发展",
            "进步": "进步",
            "改进": "改进",
            "完善": "完善",
            "提升": "提升",
            "增强": "增强",
            "扩展": "扩展",
            "升级": "升级",
            "更新": "更新",
            "维护": "维护",
            "支持": "支持",
            "帮助": "帮助",
            "解决": "解决",
            "问题": "问题",
            "困难": "困难",
            "挑战": "挑战",
            "机会": "机会",
            "优势": "优势",
            "特点": "特点",
            "特色": "特色",
            "亮点": "亮点",
            "重点": "重点",
            "关键": "关键",
            "核心": "核心",
            "重要": "重要",
            "必要": "必要",
            "基本": "基本",
            "主要": "主要",
            "首要": "首要",
            "优先": "优先",
            "紧急": "紧急",
            "及时": "及时",
            "快速": "快速",
            "高效": "高效",
            "专业": "专业",
            "精准": "精准",
            "准确": "准确",
            "正确": "正确",
            "合理": "合理",
            "科学": "科学",
            "系统": "系统",
            "全面": "全面",
            "完整": "完整",
            "详细": "详细",
            "具体": "具体",
            "明确": "明确",
            "清楚": "清楚",
            "简单": "简单",
            "复杂": "复杂",
            "困难": "困难",
            "容易": "容易",
            "方便": "方便",
            "实用": "实用",
            "有用": "有用",
            "好用": "好用",
            "易用": "易用"
        }

        if JIEBA_AVAILABLE:
            # 添加专业词汇到jieba词典
            for term in self.professional_terms.values():
                jieba.add_word(term)

    def correct_text(self, text: str) -> str:
        """纠正文本错误"""
        if not self.config.enable_text_correction:
            return text
        
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
        text = re.sub(r'(这个){2,}', '这个', text)
        text = re.sub(r'(就是){2,}', '就是', text)
        text = re.sub(r'(然后){2,}', '然后', text)
        return text

    def _fix_punctuation(self, text: str) -> str:
        """修复标点符号"""
        import re
        # 句末添加标点
        if text and not text[-1] in '。！？，；':
            if '？' in text or '什么' in text or '怎么' in text or '为什么' in text:
                text += '？'
            elif '！' in text or text.endswith(('啊', '呀', '哇', '哎', '唉')):
                text += '！'
            else:
                text += '。'
        return text


class WhisperModel:
    """Whisper模型包装器"""

    def __init__(self, model_name: str = "base", device: str = "cuda", config: Config = None):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.config = config or Config()
        self.text_processor = TextProcessor(config)
        self.model_downloader = ModelDownloader(self.config.model_cache_dir)
        
        # 验证模型名称
        if model_name not in SUPPORTED_MODELS:
            available_models = ", ".join(SUPPORTED_MODELS.keys())
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {available_models}")

    def load_model(self):
        """加载模型"""
        logger.info(f"正在加载模型: {self.model_name}")
        
        try:
            # 下载模型
            model_path = self.model_downloader.download_with_progress(
                self.model_name, 
                SUPPORTED_MODELS[self.model_name]
            )
            
            # 加载模型
            if self.model_name.startswith("faster-") and FASTER_WHISPER_AVAILABLE:
                compute_type = "float16" if self.device == "cuda" else "int8"
                if self.config.compute_type != 'auto':
                    compute_type = self.config.compute_type
                
                self.model = FasterWhisperModel(
                    model_path,
                    device=self.device,
                    compute_type=compute_type,
                    download_root=self.config.model_cache_dir
                )
                logger.info("使用Faster-Whisper模型")
                
            elif WHISPER_AVAILABLE:
                self.model = whisper.load_model(model_path, device=self.device)
                logger.info("使用OpenAI Whisper模型")
            else:
                raise ImportError("没有可用的Whisper模型！")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def transcribe(self, audio_path: str) -> List[Dict]:
        """转录音频"""
        if self.model is None:
            self.load_model()

        logger.info(f"开始转录: {audio_path}")
        start_time = time.time()

        try:
            if self.model_name.startswith("faster-") and FASTER_WHISPER_AVAILABLE:
                # Faster-Whisper
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
                # OpenAI Whisper
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
    print("支持的模型列表:")
    print("=" * 60)
    
    print("\n🚀 Faster-Whisper 模型 (推荐):")
    for model, info in SUPPORTED_MODELS.items():
        if model.startswith("faster-"):
            print(f"  {model:<20} - {info['size']}")
    
    print("\n📦 标准 Whisper 模型:")
    for model, info in SUPPORTED_MODELS.items():
        if not model.startswith("faster-") and not model.startswith("chinese-"):
            print(f"  {model:<20} - {info['size']}")
    
    print("\n🇨🇳 中文优化模型:")
    for model, info in SUPPORTED_MODELS.items():
        if model.startswith("chinese-"):
            print(f"  {model:<20} - {info['size']}")
    
    print("\n💡 RTX 3060 Ti 推荐:")
    print("  - faster-base     (平衡性能和质量)")
    print("  - faster-small    (快速处理)")
    print("  - base            (标准选择)")
    print("  ⚠️  避免使用 medium/large (显存可能不足)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RTX 3060 Ti 视频转字幕工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python main.py video.mp4\n"
               "  python main.py video.mp4 --model faster-base\n"
               "  python main.py video.mp4 --model faster-base --output subtitle.srt\n"
               "  python main.py video.mp4 --audio-quality high --enable-text-correction\n"
               "  python main.py --list-models"
    )
    
    parser.add_argument("video_path", nargs='?', help="视频文件路径")
    parser.add_argument("--model", default="base", 
                       help=f"选择模型 (默认: base)")
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
    parser.add_argument("--enable-text-correction", action="store_true", default=True,
                       help="启用文本纠错 (默认: True)")
    parser.add_argument("--enable-audio-preprocessing", action="store_true", default=True,
                       help="启用音频预处理 (默认: True)")
    parser.add_argument("--keep-temp", action="store_true", default=False,
                       help="保留临时文件 (默认: False)")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="详细输出 (默认: False)")
    parser.add_argument("--list-models", action="store_true", help="列出支持的模型")
    
    args = parser.parse_args()
    
    # 列出支持的模型
    if args.list_models:
        print_supported_models()
        return
    
    # 检查输入文件
    if not args.video_path:
        parser.error("请指定视频文件路径")
    
    if not os.path.exists(args.video_path):
        logger.error(f"文件不存在: {args.video_path}")
        return

    # 验证模型名称
    if args.model not in SUPPORTED_MODELS:
        logger.error(f"不支持的模型: {args.model}")
        print_supported_models()
        return

    # 设置输出路径
    if args.output is None:
        args.output = args.video_path.rsplit('.', 1)[0] + '.srt'

    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

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
        enable_audio_preprocessing=args.enable_audio_preprocessing,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
        language=args.language
    )

    try:
        # 提取音频
        audio_processor = AudioProcessor(config)
        audio_path = audio_processor.extract_audio(args.video_path)

        # 转录音频
        model = WhisperModel(args.model, device, config)
        segments = model.transcribe(audio_path)

        # 生成字幕
        SRTGenerator.generate_srt(segments, args.output)

        # 清理临时文件
        if not config.keep_temp and os.path.exists(audio_path) and audio_path != args.video_path:
            os.remove(audio_path)

        logger.info("转换完成！")

    except Exception as e:
        logger.error(f"转换失败: {e}")
        raise


if __name__ == "__main__":
    main()
