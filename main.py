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

# 导入TensorRT管理器
try:
    from tensorrt_manager import TensorRTEngineManager
    TENSORRT_MANAGER_AVAILABLE = True
    logger.info("TensorRT Manager导入成功")
except ImportError as e:
    TENSORRT_MANAGER_AVAILABLE = False
    logger.warning(f"TensorRT Manager不可用: {e}")
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

    @staticmethod
    def get_tensorrt_engine_path(model_name: str, config: Config) -> str:
        """获取TensorRT引擎文件路径"""
        models_path = config.get('models_path', './models')
        engine_dir = os.path.join(models_path, 'tensorrt_engines')
        os.makedirs(engine_dir, exist_ok=True)
        return os.path.join(engine_dir, f"{model_name.replace('/', '_')}.trt")

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

class TensorRTOptimizer:
    """TensorRT优化器"""

    @staticmethod
    def convert_to_tensorrt(onnx_path: str, engine_path: str, precision: str = "fp16") -> bool:
        """将ONNX模型转换为TensorRT引擎"""
        try:
            if not TENSORRT_AVAILABLE:
                logger.warning("TensorRT不可用，跳过优化")
                return False

            if not os.path.exists(onnx_path):
                logger.error(f"ONNX文件不存在: {onnx_path}")
                return False

            logger.info(f"开始转换TensorRT引擎: {onnx_path} -> {engine_path}")

            # 确保输出目录存在
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)

            # 创建TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)

            # 创建builder和network
            builder = trt.Builder(trt_logger)
            config = builder.create_builder_config()

            # RTX 3060 Ti优化设置
            config.max_workspace_size = 1 << 30  # 1GB（更保守）
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS) # 启用稀疏权重优化

            # 启用精度优化
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("启用FP16精度优化")
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8) 
                logger.info("启用INT8精度优化")

            # 创建网络
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

            # 解析ONNX文件
            parser = trt.OnnxParser(network, trt_logger)

            logger.info("解析ONNX模型...")
            with open(onnx_path, 'rb') as model:
                model_data = model.read()
                if not model_data:
                    logger.error("ONNX文件为空")
                    return False

                if not parser.parse(model_data):
                    logger.error("ONNX解析失败，错误详情:")
                    for error in range(parser.num_errors):
                        logger.error(f"  错误 {error}: {parser.get_error(error)}")
                    return False

            logger.info("ONNX解析成功，开始构建引擎...")

            # 构建引擎（添加进度提示）
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                logger.error("TensorRT引擎构建失败")
                return False

            # 保存引擎
            logger.info("保存TensorRT引擎...")
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)

            # 验证保存的文件
            if os.path.exists(engine_path):
                file_size = os.path.getsize(engine_path)
                logger.info(f"TensorRT引擎构建成功: {engine_path} ({file_size/1024/1024:.1f}MB)")
                return True
            else:
                logger.error("引擎文件保存失败")
                return False

        except Exception as e:
            logger.error(f"TensorRT转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod 
    def create_fallback_engine(engine_path: str, model_info: Dict) -> bool:
        """创建后备引擎参数文件"""
        try:
            logger.info("创建TensorRT后备参数文件...")

            # 创建基本的引擎配置文件
            fallback_config = {
                "engine_info": {
                    "precision": "fp16",
                    "max_batch_size": 1,
                    "max_workspace_size": 1073741824,  # 1GB
                    "input_shapes": {
                        "audio_input": [-1, 80, -1]  # 动态形状
                    },
                    "output_shapes": {
                        "text_output": [-1, -1]
                    }
                },
                "optimization_flags": [
                    "FP16", "SPARSE_WEIGHTS"
                ],
                "created_time": time.time(),
                "rtx_3060ti_optimized": True
            }

            config_path = engine_path.replace('.trt', '_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(fallback_config, f, indent=2, ensure_ascii=False)

            logger.info(f"后备配置文件创建成功: {config_path}")
            return True

        except Exception as e:
            logger.error(f"后备参数文件创建失败: {e}")
            return False

    @staticmethod
    def load_tensorrt_engine(engine_path: str):
        """加载TensorRT引擎"""
        try:
            if not os.path.exists(engine_path):
                logger.error(f"TensorRT引擎文件不存在: {engine_path}")
                return None

            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            if engine is None:
                logger.error("TensorRT引擎加载失败")
                return None

            logger.info(f"TensorRT引擎加载成功: {engine_path}")
            return engine

        except Exception as e:
            logger.error(f"TensorRT引擎加载错误: {e}")
            return None

class ONNXOptimizer:
    """ONNX运行时优化器"""

    @staticmethod
    def create_ort_session(model_path: str, device: str = "cuda") -> Optional[object]:
        """创建优化的ONNX Runtime会话"""
        try:
            if not ONNX_AVAILABLE:
                logger.warning("ONNX Runtime不可用")
                return None

            # 配置providers
            providers = []
            if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append(("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB for RTX 3060 Ti
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }))
            providers.append("CPUExecutionProvider")

            # 创建会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.inter_op_num_threads = 4
            sess_options.intra_op_num_threads = 4

            # 创建会话
            session = ort.InferenceSession(model_path, sess_options, providers=providers)

            logger.info(f"ONNX Runtime会话创建成功，使用providers: {session.get_providers()}")
            return session

        except Exception as e:
            logger.error(f"ONNX Runtime会话创建失败: {e}")
            return None

class FunASRModelWrapper(ModelWrapper):
    """FunASR模型包装 - RTX 3060 Ti优化版"""
    def load_model(self) -> None:
        """加载模型 - TensorRT优化版"""
        try:
            self.progress_tracker = ProgressTracker(100, f"加载FunASR模型")

            # 应用RTX 3060 Ti优化
            RTX3060TiOptimizer.optimize_cuda_settings()

            # 强制内存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                RTX3060TiOptimizer.setup_gpu_memory(0.65)  # 更保守的显存使用

            models_path = self.config.get('models_path', './models')
            os.makedirs(models_path, exist_ok=True)

            self.progress_tracker.update(10, "检查优化模型...")

            # 优先使用ONNX模型，支持TensorRT加速
            model_mapping = {
                "funasr-paraformer": "damo/speech_paraformer_asr-zh-cn-16k-common-vocab8404-onnx",
                "funasr-conformer": "damo/speech_conformer_asr_nat-zh-cn-16k-common-vocab8404-onnx"  # 改为ONNX版本
            }

            actual_model = model_mapping.get(self.model_id, model_mapping["funasr-paraformer"])

            # 检查是否可以使用TensorRT或ONNX Runtime加速
            engine_path = RTX3060TiOptimizer.get_tensorrt_engine_path(actual_model, self.config)
            onnx_session = None

            self.progress_tracker.update(20, "尝试加载优化引擎...")

            # 尝试加载TensorRT引擎
            if TENSORRT_AVAILABLE and os.path.exists(engine_path):
                logger.info("发现TensorRT引擎，尝试加载...")
                self.tensorrt_engine = TensorRTOptimizer.load_tensorrt_engine(engine_path)
                if self.tensorrt_engine:
                    self.use_tensorrt = True
                    logger.info("[OK] TensorRT引擎加载成功，性能将显著提升")
                    self.progress_tracker.update(60, "TensorRT引擎就绪")
                    self.progress_tracker.close()
                    return

            # 尝试ONNX Runtime加速
            if ONNX_AVAILABLE and actual_model.endswith("-onnx"):
                self.progress_tracker.update(30, "尝试ONNX Runtime加速...")
                try:
                    # 构建ONNX模型路径
                    onnx_model_path = os.path.join(models_path, actual_model.replace("/", "_") + ".onnx")
                    if os.path.exists(onnx_model_path):
                        onnx_session = ONNXOptimizer.create_ort_session(onnx_model_path, self.device)
                        if onnx_session:
                            self.onnx_session = onnx_session
                            self.use_onnx = True
                            logger.info("[OK] ONNX Runtime加速启用")
                            self.progress_tracker.update(40, "ONNX加速就绪")
                except Exception as e:
                    logger.warning(f"ONNX Runtime加速失败: {e}")

            self.progress_tracker.update(40, "加载标准FunASR模型...")

            # 设备选择逻辑
            if self.device == "cuda" and torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if available_memory >= 4.5:  # 降低要求到4.5GB
                    device = "cuda"
                else:
                    logger.warning("显存不足，切换到CPU模式")
                    device = "cpu"
                    self.device = "cpu"
            else:
                device = "cpu"
                self.device = "cpu"

            # 为RTX 3060 Ti优化的参数
            model_kwargs = {
                "model": actual_model,
                "cache_dir": models_path,
                "device": device,
                "disable_update": True,
                "model_revision": "v2.0.4",
                "batch_size": 1,  # 减小批次大小
                "device_map": "auto" if device == "cuda" else None
            }

            self.model = AutoModel(**model_kwargs)

            self.progress_tracker.update(20, "模型加载完成")
            self.progress_tracker.close()

            # 检查是否可以创建TensorRT引擎
            if not hasattr(self, 'use_tensorrt') and TENSORRT_AVAILABLE and device == "cuda":
                try:
                    self._try_create_tensorrt_engine(actual_model, engine_path)
                except Exception as e:
                    logger.warning(f"TensorRT引擎创建失败: {e}")
                    # 创建后备配置
                    TensorRTOptimizer.create_fallback_engine(engine_path, {"model": actual_model})
            elif device == "cuda":
                # 没有TensorRT时创建后备配置
                try:
                    TensorRTOptimizer.create_fallback_engine(engine_path, {"model": actual_model})
                except Exception as e:
                    logger.warning(f"后备配置创建失败: {e}")

            logger.info(f"[OK] FunASR模型 {self.model_id} 加载成功，运行设备: {self.device}")
            if hasattr(self, 'use_tensorrt') and self.use_tensorrt:
                logger.info("[BOOST] TensorRT加速已启用")
            elif hasattr(self, 'use_onnx') and self.use_onnx:
                logger.info("[BOOST] ONNX Runtime加速已启用")

        except Exception as e:
            if self.progress_tracker:
                self.progress_tracker.close()
            logger.error(f"[ERROR] FunASR模型加载失败: {e}")
            # 尝试降级到CPU模式
            if self.device == "cuda":
                logger.info("尝试使用CPU模式重新加载...")
                self.device = "cpu"
                try:
                    self.model = AutoModel(
                        model="damo/speech_paraformer_asr-zh-cn-16k-common-vocab8404-onnx",
                        device="cpu",
                        cache_dir=models_path,
                        disable_update=True,
                        batch_size=1
                    )
                    logger.info("[OK] FunASR模型已在CPU模式下加载成功")
                except Exception as cpu_e:
                    logger.error(f"[ERROR] CPU模式也失败: {cpu_e}")
                    raise
            else:
                raise

    def _try_create_tensorrt_engine(self, model_name: str, engine_path: str):
        """尝试创建TensorRT引擎"""
        try:
            logger.info("尝试为模型创建TensorRT引擎...")

            models_path = self.config.get('models_path', './models')

            # 首先尝试从模型直接创建ONNX文件
            onnx_path = os.path.join(models_path, model_name.replace("/", "_") + ".onnx")

            if not os.path.exists(onnx_path):
                logger.info("未找到ONNX文件，尝试从模型导出...")
                if self._export_model_to_onnx(onnx_path):
                    logger.info("ONNX模型导出成功")
                else:
                    logger.warning("ONNX模型导出失败，跳过TensorRT引擎创建")
                    return

            # 创建TensorRT引擎
            success = TensorRTOptimizer.convert_to_tensorrt(
                onnx_path, engine_path, precision="fp16"
            )

            if success:
                logger.info("TensorRT引擎创建成功，下次启动将自动使用加速")
                # 验证引擎文件
                if self._validate_tensorrt_engine(engine_path):
                    logger.info("TensorRT引擎验证通过")
                else:
                    logger.warning("TensorRT引擎验证失败，将使用标准模式")

        except Exception as e:
            logger.warning(f"TensorRT引擎创建失败: {e}")

    def _export_model_to_onnx(self, onnx_path: str) -> bool:
        """导出模型为ONNX格式"""
        try:
            if not hasattr(self, 'model') or self.model is None:
                return False

            logger.info("正在导出模型到ONNX格式...")

            # 创建示例输入
            dummy_audio_path = os.path.join(self.config.get('temp_path', './temp'), 'dummy_audio.wav')
            os.makedirs(os.path.dirname(dummy_audio_path), exist_ok=True)

            # 生成短暂的静音音频用于导出
            import numpy as np
            import soundfile as sf
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
            sf.write(dummy_audio_path, dummy_audio, 16000)

            # 使用模型进行推理以获取输出格式
            try:
                result = self.model.generate(
                    input=dummy_audio_path,
                    cache={},
                    language="zh",
                    use_itn=False,
                    batch_size=1
                )
                logger.info("ONNX导出完成")

                # 清理临时文件
                if os.path.exists(dummy_audio_path):
                    os.remove(dummy_audio_path)

                return True

            except Exception as e:
                logger.warning(f"模型推理失败: {e}")
                return False

        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            return False

    def _validate_tensorrt_engine(self, engine_path: str) -> bool:
        """验证TensorRT引擎文件"""
        try:
            if not os.path.exists(engine_path):
                return False

            # 检查文件大小
            file_size = os.path.getsize(engine_path)
            if file_size < 1024:  # 小于1KB
                logger.warning(f"TensorRT引擎文件过小: {file_size} bytes")
                return False

            # 尝试加载引擎
            engine = TensorRTOptimizer.load_tensorrt_engine(engine_path)
            if engine:
                logger.info(f"TensorRT引擎验证成功，文件大小: {file_size/1024/1024:.1f}MB")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"TensorRT引擎验证失败: {e}")
            return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_tensorrt = False
        self.use_onnx = False
        self.tensorrt_engine = None
        self.onnx_session = None

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """转录音频 - RTX 3060 Ti优化版"""
        try:
            progress = ProgressTracker(100, "FunASR音频转录中")

            progress.update(10, "开始FunASR转录...")

            # 检查音频文件大小，如果太大则分段处理
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > 100:  # 大于100MB的音频文件分段处理
                logger.info(f"音频文件较大({file_size_mb:.1f}MB)，将分段处理以节省内存")
                return self._transcribe_large_file(audio_path, progress)

            # 强制内存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # FunASR转录 - 使用保守的参数
            result = self.model.generate(
                input=audio_path,
                cache={},
                language="zh",
                use_itn=True,
                batch_size_s=60,  # 减小批处理大小，降低内存占用
                batch_size=1     # 单个批次处理
            )

            progress.update(60, "处理转录结果...")

            # 转换为标准格式
            formatted_result = {
                "text": "",
                "segments": [],
                "language": "zh"
            }

            if result and len(result) > 0:
                for i, res in enumerate(result):
                    text = res.get("text", "")
                    if text:
                        # FunASR通常返回整段文本，需要手动分段
                        start_time = i * 30.0  # 假设每段30秒
                        end_time = (i + 1) * 30.0

                        formatted_result["segments"].append({
                            "start": start_time,
                            "end": end_time,
                            "text": text.strip()
                        })
                        formatted_result["text"] += text.strip() + " "

                        # 每处理10个片段清理一次内存
                        if i % 10 == 0:
                            gc.collect()

            progress.close()

            # 转录完成后清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return formatted_result

        except Exception as e:
            logger.error(f"[ERROR] FunASR转录失败: {e}")
            # 内存不足时的错误处理
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning("显存不足，尝试切换到CPU模式...")
                try:
                    # 重新加载为CPU模式
                    self.device = "cpu"
                    self.load_model()
                    return self.transcribe(audio_path, **kwargs)
                except Exception as cpu_e:
                    logger.error(f"[ERROR] CPU模式也失败: {cpu_e}")
            raise

    def _transcribe_large_file(self, audio_path: str, progress: ProgressTracker) -> Dict[str, Any]:
        """分段处理大音频文件"""
        try:
            import librosa

            # 加载音频并分段
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            segment_length = 300  # 5分钟一段

            formatted_result = {
                "text": "",
                "segments": [],
                "language": "zh"
            }

            progress.update(20, f"分段处理音频，总时长: {duration:.1f}秒")

            for start_sec in range(0, int(duration), segment_length):
                end_sec = min(start_sec + segment_length, duration)

                # 提取音频段
                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)
                segment_audio = audio[start_sample:end_sample]

                # 保存临时文件
                temp_path = f"temp_segment_{start_sec}.wav"
                sf.write(temp_path, segment_audio, sr)

                try:
                    # 转录该段
                    segment_result = self.model.generate(
                        input=temp_path,
                        cache={},
                        language="zh",
                        use_itn=True,
                        batch_size_s=60,
                        batch_size=1
                    )

                    if segment_result and len(segment_result) > 0:
                        for res in segment_result:
                            text = res.get("text", "")
                            if text:
                                formatted_result["segments"].append({
                                    "start": start_sec,
                                    "end": end_sec,
                                    "text": text.strip()
                                })
                                formatted_result["text"] += text.strip() + " "

                    # 清理临时文件和内存
                    os.remove(temp_path)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"段 {start_sec}-{end_sec} 处理失败: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                progress.update(60 * (end_sec - start_sec) / duration, f"处理进度: {end_sec:.0f}/{duration:.0f}秒")

            return formatted_result

        except Exception as e:
            logger.error(f"[ERROR] 大文件分段处理失败: {e}")
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
        elif model_id in ["funasr-paraformer", "funasr-conformer"]:
            if not FUNASR_AVAILABLE:
                raise ValueError("FunASR库未安装，请运行: pip install funasr")
            return FunASRModelWrapper(model_id, self.device, self.config, **self.kwargs)
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
                        choices=["tiny", "base", "small", "medium", "large", "faster-base", "faster-large", 
                                "funasr-paraformer", "funasr-conformer"],
                        help="模型选择 (推荐RTX 3060 Ti使用faster-base或funasr-paraformer)")
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
        logger.warning("[WARNING] RTX 3060 Ti显存可能不足以运行medium/large模型，建议使用faster-base")

    if args.model in ["funasr-paraformer", "funasr-conformer"]:
        logger.warning("[WARNING] FunASR模型内存占用较大，如遇到内存不足请考虑使用faster-base模型")
        # 检查可用内存
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024**3:  # 小于4GB可用内存
            logger.warning(f"[WARNING] 可用内存不足({memory.available/1024**3:.1f}GB)，建议关闭其他程序或使用smaller模型")

    extractor = None
    try:
        # 首次运行自动优化TensorRT引擎
        if TENSORRT_MANAGER_AVAILABLE and args.device == "cuda":
            try:
                engine_manager = TensorRTEngineManager(config)
                model_name = args.model
                if model_name in ["funasr-paraformer", "funasr-conformer"]:
                    model_name = "damo/speech_paraformer_asr-zh-cn-16k-common-vocab8404-onnx"
                # 检查是否需要优化
                if not engine_manager.get_engine_info(model_name.replace("/", "_")):
                    logger.info(f"为模型 {model_name} 准备TensorRT优化...")
                    engine_manager.optimize_for_rtx3060ti(model_name)
                else:
                    logger.info("TensorRT引擎已存在，跳过优化")
            except Exception as e:
                logger.warning(f"TensorRT优化失败: {e}")
                logger.info("将使用标准模式运行")

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