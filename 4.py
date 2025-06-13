import os
import time
import torch
import argparse
import traceback
import subprocess
import platform
from datetime import timedelta
from typing import List, Dict, Any, Optional
import logging
import sys
import re
import jieba
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
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
os.environ['CUDA_LAZY_LOADING'] = '1'
# 尝试导入TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    from faster_whisper import WhisperModel
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    TENSORRT_AVAILABLE = True
    logger.info("TensorRT导入成功")
except ImportError as e:
    TENSORRT_AVAILABLE = False
    logger.warning(f"TensorRT或相关依赖未安装: {e}，将使用默认推理模式")
    try:
        from faster_whisper import WhisperModel
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as e:
        logger.error(f"必要的模型库未安装: {e}")
        sys.exit(1)

# 启用PyTorch显存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 新增FireRedASR导入
try:
    from fireredasr.models.fireredasr import FireRedAsr
    logger.info("FireRedASR库导入成功")
except ImportError:
    logger.warning("未找到FireRedASR库，请确保已安装: pip install fireredasr")
    FireRedAsr = None

# 设置FFmpeg路径（根据您的实际路径修改）
os.environ["PATH"] += os.pathsep + r"D:\code\ffmpeg\bin"

# 添加CUDA路径到系统PATH
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
if os.path.exists(cuda_path) and cuda_path not in os.environ["PATH"]:
    os.environ["PATH"] = cuda_path + ";" + os.environ["PATH"]
    logger.info(f"已将CUDA路径添加到系统PATH: {cuda_path}")

# 修改 ModelWrapper 基类
class ModelWrapper(torch.nn.Module):
    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.kwargs = kwargs
        self.model = None  # Will be set by child classes
        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp_audio")
        os.makedirs(self.temp_dir, exist_ok=True)

    def forward(self, x):
        # 调整输入维度：(batch, seq, feat) → (batch, feat, seq)
        x = x.permute(0, 2, 1)
        x_lengths = torch.tensor([x.size(2)], device=x.device)  # 序列长度

        # 使用模型的 transcribe 方法进行推理
        if hasattr(self.model, 'transcribe'):
            try:
                # 创建临时文件来存储音频数据
                temp_path = os.path.join(self.temp_dir, "temp_input.wav")

                # 将张量转换为numpy数组并保存为WAV文件
                audio_data = x.cpu().numpy()
                sf.write(temp_path, audio_data, 16000)

                # 调用transcribe方法，使用正确的参数
                results = self.model.transcribe(
                    batch_uttid=["single"],
                    batch_wav_path=[temp_path],
                    args={
                        "gpu": 1 if x.is_cuda else 0,
                        "batch_size": 1,
                        "compute_type": "float16" if x.dtype == torch.float16 else "float32"
                    }
                )

                # 清理临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass

                # 提取文本结果
                if results and isinstance(results, list) and len(results) > 0:
                    text = results[0]["text"]
                    # 将文本转换为张量
                    return torch.tensor([ord(c) for c in text], device=x.device)
                else:
                    return torch.zeros(1, device=x.device)

            except Exception as e:
                logger.error(f"Forward pass failed: {e}")
                return torch.zeros(1, device=x.device)
        else:
            raise ValueError("模型不支持transcribe方法")


# Whisper模型包装类
class WhisperModelWrapper(ModelWrapper):
    def load_model(self):
        model_size = self.model_id.replace("whisper-", "")  # 解析model_id格式如whisper-large-v3
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


# Hugging Face模型包装类
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

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=4096,
            chunk_length_s=30,
            batch_size=4,
            torch_dtype=torch_dtype,
            device=self.device,
        )

        logger.info(f"Hugging Face模型 {self.model_id} 加载完成")

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        try:
            transcription_args = {
                "return_timestamps": True,
                "generate_kwargs": {
                    "temperature": kwargs.get("temperature", 0.2),
                    "num_beams": kwargs.get("beam_size", 5),
                }
            }

            if "belle-whisper" in self.model_id or kwargs.get("language") == "zh":
                if hasattr(self.processor, "get_decoder_prompt_ids"):
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language="zh", task="transcribe"
                    )
                    transcription_args["generate_kwargs"]["forced_decoder_ids"] = forced_decoder_ids
                    logger.info("已启用中文强制解码")

            if kwargs.get("language"):
                transcription_args["generate_kwargs"]["language"] = kwargs["language"]

            if self.model_id.startswith("openai/whisper") or "belle-whisper" in self.model_id:
                transcription_args["generate_kwargs"]["task"] = "transcribe"

            result = self.pipe(audio_path, **transcription_args)

            segments = []
            if "chunks" in result:
                for chunk in result["chunks"]:
                    segments.append({
                        "start": chunk["timestamp"][0],
                        "end": chunk["timestamp"][1],
                        "text": chunk["text"]
                    })
            else:
                duration = self.get_audio_duration(audio_path)
                segments.append({
                    "start": 0.0,
                    "end": duration,
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
            command = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            return float(result.stdout.strip()) if result.stdout else 0.0


# 优化后的 TensorRT Engine 类（用于 Windows 更稳定执行）
import os
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TensorRTEngine:
    def __init__(self, model_path: str, device: str = "cuda", precision: str = "fp16", model=None):
        self.model_path = os.path.abspath(model_path)
        self.device = device
        self.precision = precision
        self.model = model
        self.engine = None
        self.context = None
        self.stream = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self._build_or_load()

    def _build_or_load(self):
        engine_file = os.path.join(self.model_path, "model.trt")
        if os.path.exists(engine_file):
            self._load_engine(engine_file)
        else:
            self._build_engine(engine_file)

    def _build_engine(self, engine_file):
        from torch import nn
        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)
        config = builder.create_builder_config()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        if self.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        onnx_file = os.path.join(self.model_path, "temp_model.onnx")

        wrapped_model = self._wrap_model(self.model)
        dummy_input = torch.randn(1, 500, 80).to(self.device)
        if self.precision == "fp16":
            wrapped_model = wrapped_model.half()
            dummy_input = dummy_input.half()

        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_file,
            export_params=True,
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None
        )

        parser = trt.OnnxParser(network, trt_logger)
        with open(onnx_file, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX 解析错误: {parser.get_error(i)}")
                raise RuntimeError("TensorRT ONNX 解析失败")

        engine = builder.build_engine(network, config)
        with open(engine_file, 'wb') as f:
            f.write(engine.serialize())
        os.remove(onnx_file)
        self._load_engine(engine_file)

    def _load_engine(self, engine_file):
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            with open(engine_file, 'rb') as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError("TensorRT引擎反序列化失败")

            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            # 清空之前的绑定
            self.inputs.clear()
            self.outputs.clear()
            self.bindings.clear()

            for binding in range(self.engine.num_bindings):
                # shape = self.engine.get_binding_shape(binding)
                # size = trt.volume(shape)
                # dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                name = self.engine.get_binding_name(binding)
                shape = self.engine.get_tensor_shape(name)
                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))

                # 确保size > 0
                if size <= 0:
                    logger.warning(f"绑定 {binding} 的大小无效: {size}")
                    continue

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))

                # if self.engine.binding_is_input(binding):
                #     self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                #     logger.info(f"输入绑定 {binding}: shape={shape}, size={size}")
                # else:
                #     self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                #     logger.info(f"输出绑定 {binding}: shape={shape}, size={size}")
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                    logger.info(f"输入绑定 {binding}: shape={shape}, size={size}")
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                    logger.info(f"输出绑定 {binding}: shape={shape}, size={size}")

            if not self.inputs:
                raise RuntimeError("TensorRT引擎没有输入绑定")
            if not self.outputs:
                raise RuntimeError("TensorRT引擎没有输出绑定")

            logger.info(f"TensorRT引擎加载成功，输入数量: {len(self.inputs)}, 输出数量: {len(self.outputs)}")

        except Exception as e:
            logger.error(f"TensorRT引擎加载失败: {e}")
            raise

    def _wrap_model(self, model):
        import torch.nn as nn
        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                x = x.permute(0, 2, 1)
                # 创建临时文件来存储音频数据
                temp_path = os.path.join(os.path.dirname("./"), "temp_input.wav")
                try:
                    # 将张量转换为numpy数组并保存为WAV文件
                    audio_data = x.cpu().numpy()
                    sf.write(temp_path, audio_data, 16000)

                    # 调用transcribe方法，使用正确的参数
                    out = self.m.transcribe(
                        batch_uttid=["single"],
                        batch_wav_path=[temp_path],
                        args={"gpu": 1, "compute_type": "float16"}
                    )

                    # 清理临时文件
                    try:
                        os.remove(temp_path)
                    except:
                        pass

                    # 提取文本结果
                    t = out[0]["text"] if isinstance(out, list) else out["text"]
                    return torch.tensor([ord(c) for c in t]).to(x.device)
                except Exception as e:
                    logger.error(f"Forward pass failed: {e}")
                    return torch.zeros(1, device=x.device)

        return Wrapper(model)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        try:
            if not self.inputs:
                raise RuntimeError("TensorRT引擎未正确初始化，没有输入绑定")
            if not self.outputs:
                raise RuntimeError("TensorRT引擎未正确初始化，没有输出绑定")

            # 确保输入数据形状匹配
            input_shape = self.inputs[0]['shape']
            flattened_input = input_data.ravel()
            expected_size = np.prod(input_shape)

            if len(flattened_input) != expected_size:
                logger.warning(f"输入数据大小不匹配，期望: {expected_size}, 实际: {len(flattened_input)}")
                # 调整数据大小
                if len(flattened_input) > expected_size:
                    flattened_input = flattened_input[:expected_size]
                else:
                    padded_input = np.zeros(expected_size, dtype=flattened_input.dtype)
                    padded_input[:len(flattened_input)] = flattened_input
                    flattened_input = padded_input

            # 复制数据到主机内存
            np.copyto(self.inputs[0]['host'], flattened_input)

            # 传输数据到设备
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

            # 执行推理
            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not success:
                raise RuntimeError("TensorRT推理执行失败")

            # 传输结果回主机
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # 重塑输出
            output_shape = self.outputs[0]['shape']
            return self.outputs[0]['host'].reshape(output_shape)

        except Exception as e:
            logger.error(f"TensorRT推理失败: {e}")
            raise

    def __del__(self):
        try:
            if self.stream:
                self.stream.synchronize()
            if self.context:
                self.context = None
            if self.engine:
                self.engine = None
            for output in self.outputs:
                if 'device' in output:
                    output['device'].free()
        except Exception as e:
            logger.warning(f"TensorRT资源清理失败: {e}")

    def get_gpu_memory_usage(self) -> float:
        """获取当前GPU显存使用量（MB）"""
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.memory_allocated() / (1024 ** 2)
        except Exception as e:
            logger.warning(f"获取GPU显存使用量失败: {e}")
            return 0.0


# 优化后的FireRedASR模型包装类
class FireRedASRWrapper(ModelWrapper):
    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        super().__init__(model_id, device, **kwargs)
        self.model_type = model_id.split("-")[1]
        self.model_path = kwargs.get("model_path", f"pretrained_models/FireRedASR-{self.model_type.upper()}-L")
        self._validate_model_type()
        self.max_segment_length = kwargs.get("max_segment_length", 30)
        self.segment_overlap = kwargs.get("segment_overlap", 5)
        self.max_gpu_memory = kwargs.get("max_gpu_memory", 3500)
        self.enable_memory_optimization = True
        self.use_half_precision = False
        self.memory_cleanup_threshold = 0.85
        self.temp_dir = kwargs.get("temp_dir", "temp_audio")
        self.batch_size = kwargs.get("batch_size", 1)
        self.num_workers = kwargs.get("num_workers", 1)
        self.use_tensorrt = kwargs.get("use_tensorrt", False) and TENSORRT_AVAILABLE
        self.tensorrt_engine = None
        self._ensure_temp_dir()

    def _ensure_temp_dir(self):
        """确保临时目录存在并清空"""
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")
        else:
            os.makedirs(self.temp_dir)

    def _get_temp_path(self, prefix: str, index: int) -> str:
        """获取临时文件路径"""
        return os.path.join(self.temp_dir, f"{prefix}_{index}.wav")

    def _merge_segments(self, segments: List[Dict], overlap: float = 5.0) -> List[Dict]:
        """合并重叠的片段"""
        if not segments:
            return segments

        merged = []
        current = segments[0].copy()

        for next_seg in segments[1:]:
            # 检查是否有重叠
            if next_seg["start"] - current["end"] < overlap:
                # 合并文本
                current["text"] = current["text"].rstrip() + " " + next_seg["text"].lstrip()
                current["end"] = next_seg["end"]
            else:
                merged.append(current)
                current = next_seg.copy()

        merged.append(current)
        return merged

    def _validate_model_type(self) -> None:
        """验证模型类型是否为支持的AED或LLM"""
        if self.model_type not in ["aed", "llm"]:
            raise ValueError(f"FireRedASR模型类型必须为aed或llm，得到: {self.model_type}")

    def _force_memory_cleanup(self):
        """强制清理显存"""
        if torch.cuda.is_available():
            try:
                # 清理缓存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # 清理未使用的内存
                gc.collect()

                # 重置CUDA设备
                torch.cuda.reset_peak_memory_stats()

                # 等待一小段时间让CUDA完成清理
                time.sleep(0.1)

                logger.info(f"执行强制显存清理，当前显存占用: {self.get_gpu_memory_usage()}MB")
            except Exception as e:
                logger.warning(f"显存清理失败: {e}")

    def _check_memory_usage(self):
        """检查显存使用情况"""
        if not torch.cuda.is_available():
            return True

        try:
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

            usage_ratio = memory_allocated / total_memory

            if usage_ratio > self.memory_cleanup_threshold:
                logger.warning(f"显存使用率过高 ({usage_ratio:.2%})，执行清理...")
                self._force_memory_cleanup()
                return False
            return True
        except Exception as e:
            logger.warning(f"显存检查失败: {e}")
            return False

    def load_model(self) -> None:
        """加载模型并进行优化"""
        try:
            os.environ["FIREDASR_MODEL_DIR"] = self.model_path

            # 尝试初始化TensorRT（可选优化）
            if self.use_tensorrt and TENSORRT_AVAILABLE:
                try:
                    logger.info("尝试为FireRedASR初始化TensorRT引擎...")

                    # 首先加载PyTorch模型用于转换
                    if not hasattr(self, 'model') or self.model is None:
                        logger.info("加载PyTorch模型用于TensorRT转换...")
                        self.model = FireRedAsr.from_pretrained(self.model_type)

                        if self.device == "cuda" and torch.cuda.is_available():
                            self.model.model = self.model.model.to(self.device)

                    # 尝试构建TensorRT引擎
                    self.tensorrt_engine = TensorRTEngine(
                        self.model_path,
                        self.device,
                        precision=self.kwargs.get("tensorrt_precision", "fp16"),
                        model=self.model
                    )
                    logger.info("TensorRT引擎初始化成功，将使用TensorRT加速推理")

                except Exception as e:
                    logger.warning(f"TensorRT初始化失败，回退到PyTorch模型: {e}")
                    self.use_tensorrt = False
                    self.tensorrt_engine = None

            # 如果没有使用TensorRT，加载标准PyTorch模型
            if not self.use_tensorrt:
                logger.info("使用PyTorch模型")

                # 避免重复加载模型
                if not hasattr(self, 'model') or self.model is None:
                    self.model = FireRedAsr.from_pretrained(self.model_type)

                if self.device == "cuda" and torch.cuda.is_available():
                    self._force_memory_cleanup()
                    self.model.model = self.model.model.to(self.device)

                    if self.enable_memory_optimization:
                        if hasattr(self.model, 'batch_size'):
                            self.model.batch_size = self.batch_size

                        if hasattr(self.model.model, 'encoder'):
                            self.model.model.encoder.use_checkpoint = True
                            self.model.model.encoder.parallel = True

                        self._force_memory_cleanup()

                    logger.info(f"GPU优化完成，当前显存占用: {self.get_gpu_memory_usage()}MB")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            traceback.print_exc()
            raise

    def _process_audio_batch(self, audio_files: List[str], decode_params: Dict[str, Any] = None) -> List[Dict]:
        """批量处理音频文件"""
        results = []
        try:
            # 总是使用PyTorch模型进行推理（更稳定）
            batch_uttids = [f"utt_{i}" for i in range(len(audio_files))]

            logger.info(f"处理 {len(audio_files)} 个音频文件")

            batch_results = self.model.transcribe(
                batch_uttid=batch_uttids,
                batch_wav_path=audio_files,
                args=decode_params or {}
            )

            for i, result in enumerate(batch_results):
                duration = self.get_audio_duration(audio_files[i])
                results.append({
                    "segments": [{
                        "start": 0.0,
                        "end": duration,
                        "text": result.get("text", "")
                    }],
                    "language": "zh"
                })

        except Exception as e:
            logger.error(f"批处理转录失败: {e}")
            traceback.print_exc()

            # 返回空结果而不是完全失败
            for audio_file in audio_files:
                duration = self.get_audio_duration(audio_file)
                results.append({
                    "segments": [{
                        "start": 0.0,
                        "end": duration,
                        "text": ""
                    }],
                    "language": "zh"
                })

        return results

    def _load_audio_data(self, audio_path: str) -> np.ndarray:
        """加载音频数据为numpy数组"""
        try:
            import librosa
            audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)
            return audio_data.astype(np.float32)
        except Exception as e:
            logger.error(f"音频数据加载失败: {e}")
            raise

    def _decode_output(self, output: np.ndarray) -> str:
        """解码TensorRT输出"""
        # 这里需要根据模型的输出格式进行解码
        # 示例实现，需要根据实际模型调整
        return "".join([chr(int(x)) for x in output if x > 0])

    def transcribe(self, audio_path: str, **kwargs: Any) -> Dict[str, Any]:
        """音频转录主逻辑，支持长音频分块处理"""
        try:
            # 构建解码参数
            decode_params = self._build_decode_params(kwargs)

            # 获取音频时长
            audio_duration = self.get_audio_duration(audio_path)

            # 根据音频长度决定是否分块处理
            if audio_duration > 300:  # 超过5分钟自动分块
                return self._chunked_transcribe(audio_path, decode_params, audio_duration)

            return self._single_chunk_transcribe(audio_path, decode_params, audio_duration)
        except Exception as e:
            logger.error(f"转录失败: {e}")
            return {"segments": [], "language": "zh"}

    def _build_decode_params(self, kwargs: Any) -> Dict[str, Any]:
        """构建解码参数"""
        # 基础参数
        base_params = {
            "gpu": 1 if self.device == "cuda" else 0,
            "beam_size": 3,
            "nbest": 1,
            "compute_type": "float16",  # 强制FP16计算
            "max_batch_size": 1,  # 单例推理
            "chunk_size": 16000  # 1秒音频采样点（16kHz）
        }

        # 从kwargs中获取decode_params
        decode_params = kwargs.get("decode_params", {})

        # 只保留模型支持的参数
        supported_params = {
            "gpu", "beam_size", "nbest", "compute_type",
            "max_batch_size", "chunk_size", "decode_max_len",
            "decode_min_len", "repetition_penalty", "llm_length_penalty"
        }

        # 过滤掉不支持的参数
        filtered_params = {
            k: v for k, v in decode_params.items()
            if k in supported_params
        }

        # 合并参数
        base_params.update(filtered_params)
        return base_params

    def _single_chunk_transcribe(
        self, audio_path: str, decode_params: Dict[str, Any], duration: float
    ) -> Dict[str, Any]:
        """单块转录"""
        try:
            # 加载音频数据
            audio, sr = sf.read(audio_path, dtype="float32")
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()

            # 准备输入
            batch_uttid = ["single"]
            batch_wav_path = [audio_path]

            # 执行转录
            results = self.model.transcribe(
                batch_uttid=batch_uttid,
                batch_wav_path=batch_wav_path,
                args=decode_params
            )

            return self._parse_results(results, duration)
        except Exception as e:
            logger.error(f"单块转录失败: {e}")
            return {"segments": [], "language": "zh", "duration": duration}

    def _parse_results(self, results: List[Dict], audio_path: str) -> Dict[str, Any]:
        """解析转录结果"""
        if not results:
            return {"segments": [], "language": "zh"}

        # 获取第一个结果（单个音频）        result = results[0]
        duration = self.get_audio_duration(audio_path)

        return {
            "segments": [{
                "start": 0.0,
                "end": duration,
                "text": result["text"]
            }],
            "language": "zh"
        }

    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            command = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            return float(result.stdout.strip()) if result.stdout else 0.0
        except:
            return 0.0

    def _extract_audio_segment(self, audio_path: str, start: float, end: float, output_path: str) -> None:
        """提取音频片段"""
        command = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ss", str(start),
            "-to", str(end),
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            output_path
        ]

        subprocess.run(command, check=True, capture_output=True)

    def _chunked_transcribe(self, audio_path: str, decode_params: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """分块转录"""
        try:
            segments = []
            start_time = 0.0
            segment_index = 0
            batch_files = []

            # 使用tqdm显示分块转录进度
            with tqdm(total=duration, unit="秒", desc="分块转录") as pbar:
                while start_time < duration:
                    if not self._check_memory_usage():
                        logger.warning("显存不足，等待清理...")
                        time.sleep(1)  # 增加等待时间
                        continue

                    segment_index += 1
                    end_time = min(start_time + self.max_segment_length, duration)
                    segment_file = self._get_temp_path("segment", segment_index)

                    # 提取音频片段
                    self._extract_audio_segment(audio_path, start_time, end_time, segment_file)
                    batch_files.append(segment_file)

                    # 当达到批处理大小时进行处理
                    if len(batch_files) >= self.batch_size or end_time >= duration:
                        logger.info(f"处理音频分段 {segment_index-len(batch_files)+1}-{segment_index}")

                        try:
                            # 批量处理音频
                            batch_results = self._process_audio_batch(batch_files, decode_params)

                            # 处理结果
                            for i, result in enumerate(batch_results):
                                current_start = start_time - (len(batch_files) - i - 1) * (self.max_segment_length - self.segment_overlap)
                                for seg in result["segments"]:
                                    seg["start"] += current_start
                                    seg["end"] += current_start
                                segments.extend(result["segments"])

                            # 清理已处理的文件
                            for file in batch_files:
                                try:
                                    os.remove(file)
                                except Exception as e:
                                    logger.warning(f"删除临时文件失败: {file} - {e}")

                            batch_files = []

                            # 每次批处理后强制清理显存
                            self._force_memory_cleanup()

                        except Exception as e:
                            logger.error(f"批处理失败: {e}")
                            self._force_memory_cleanup()
                            continue

                    # 更新进度条
                    pbar.update(end_time - start_time)
                    start_time = end_time - self.segment_overlap

                # 合并重叠的片段
                merged_segments = self._merge_segments(segments, self.segment_overlap)
                return {"segments": merged_segments, "language": "zh"}

        except Exception as e:
            logger.error(f"分段转录失败: {e}")
            traceback.print_exc()
            return {"segments": [], "language": "zh"}

    def get_gpu_memory_usage(self) -> float:
        """获取当前GPU显存使用量（MB）"""
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.memory_allocated() / (1024 ** 2)
        except Exception as e:
            logger.warning(f"获取GPU显存使用量失败: {e}")
            return 0.0


# 模型工厂类
class ModelFactory:
    MODEL_MAPPING = {
        "whisper": WhisperModelWrapper,  # faster-whisper
        "hf-whisper": HuggingFaceModelWrapper,  # Hugging Face Whisper系列
        "hf-belle-whisper": HuggingFaceModelWrapper,  # BELLE等第三方Whisper变体
        "hf": HuggingFaceModelWrapper,  # 通用Hugging Face ASR模型
        "fireredasr": FireRedASRWrapper,  # FireRedASR模型
    }

    @staticmethod
    def create_model(model_id: str, **kwargs) -> ModelWrapper:
        # 检测FireRedASR模型
        if model_id.startswith("fireredasr-"):
            return ModelFactory.MODEL_MAPPING["fireredasr"](model_id, **kwargs)

        # 原有模型检测逻辑
        elif model_id.startswith("hf-belle-") or "belle-whisper" in model_id:
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
            # 更轻量的标准化方法
            command = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "loudnorm=I=-23:LRA=7",
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                "-loglevel", "error", output_path
            ]
            subprocess.run(command, check=True)
            logger.info("FFmpeg轻量音量标准化完成")
            return output_path
        except Exception as e:
            logger.error(f"音量标准化失败: {e}")
            return audio_path

    @staticmethod
    def denoise_audio(audio_path, output_path):
        try:
            # 使用更通用的降噪方法，不依赖外部模型文件
            command = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "highpass=f=200,lowpass=f=3000,afftdn=nf=-25",
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                "-loglevel", "error", output_path
            ]
            subprocess.run(command, check=True)
            logger.info("降噪处理完成")
            return output_path
        except Exception as e:
            logger.error(f"降噪处理失败: {e}")
            logger.info("跳过降噪处理，使用原始音频")
            return audio_path


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


# 视频字幕提取器类
class VideoSubtitleExtractor:
    def __init__(self, model_id: str = "whisper-large-v3", device: str = "cuda", **kwargs):
        self.device = device
        self.model_id = model_id
        self.kwargs = kwargs
        SystemMonitor.check_environment()

        # 针对BELLE模型调整默认参数
        if "belle-whisper" in model_id:
            self.kwargs.update({
                "beam_size": 15,
                "temperature": 0.1
            })
            logger.info(f"BELLE模型参数优化: beam_size=15, temperature=0.1")

        # 针对FireRedASR模型的特殊处理
        if model_id.startswith("fireredasr-"):
            model_type = model_id.split("-")[1]
            if model_type not in ["aed", "llm"]:
                raise ValueError(f"FireRedASR模型类型必须为aed或llm，得到: {model_type}")

            # 设置默认模型路径
            if "model_path" not in self.kwargs:
                self.kwargs["model_path"] = os.path.expanduser(f"~/.cache/modelscope/hub/pengzhendong/FireRedASR-{model_type.upper()}-L")

            logger.info(f"使用FireRedASR {model_type.upper()} 模型，路径: {self.kwargs['model_path']}")

        # 初始化模型
        self.model_wrapper = ModelFactory.create_model(model_id, device=device, **kwargs)
        self.model_wrapper.load_model()

    def extract_audio(self, video_path):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = f"{base_name}_extracted_audio.wav"
        command = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",  # 降低采样率到16kHz
            "-acodec", "pcm_s16le", "-loglevel", "error",
            audio_path
        ]
        subprocess.run(command, check=True)
        logger.info(f"音频提取完成: {audio_path}")
        return audio_path

    def preprocess_audio(self, audio_path, denoise=True):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        processed_path = f"{base_name}_processed.wav"

        # 音量标准化
        normalized_path = AudioPreprocessor.normalize_audio(audio_path, processed_path)

        # 如果需要降噪
        if denoise:
            denoised_path = AudioPreprocessor.denoise_audio(normalized_path, processed_path)
            return denoised_path
        return normalized_path

    def transcribe_audio(self, audio_path, language=None, temperature=0.2):
        logger.info("开始音频转录...")

        # 获取音频信息
        duration = self.get_audio_duration(audio_path)
        logger.info(f"音频时长: {duration:.2f} 秒")

        options = {
            "language": language,
            "temperature": temperature,
            "vad_filter": True,
            "beam_size": 10,
            "initial_prompt": "以下是普通话对话，包含日常用语和专有名词。"
        }

        # 针对FireRedASR模型的特殊参数处理
        if self.model_id.startswith("fireredasr-"):
            decode_params = {
                "use_gpu": 1 if self.device == "cuda" else 0,
                "temperature": temperature,
                "beam_size": options["beam_size"]
            }

            # 根据模型类型设置特定参数
            model_type = self.model_id.split("-")[1]
            if model_type == "aed":
                decode_params.update({
                    "nbest": 1,
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.0,
                    "aed_length_penalty": 0.0,
                    "eos_penalty": 1.0
                })
            elif model_type == "llm":
                decode_params.update({
                    "decode_max_len": 0,
                    "decode_min_len": 0,
                    "repetition_penalty": 1.0,
                    "llm_length_penalty": 0.0
                })

            options = {"decode_params": decode_params}

        result = self.model_wrapper.transcribe(audio_path, **options)

        # 中文文本处理
        if result["language"] == "zh":
            for seg in result["segments"]:
                seg["text"] = ChineseTextProcessor.enhance_chinese_text(seg["text"])
        return result

    def create_srt_file(self, segments, output_path="output.srt"):
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                if hasattr(seg, 'start'):
                    start = seg.start
                    end = seg.end
                    text = seg.text
                else:
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

    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            command = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            return float(result.stdout.strip()) if result.stdout else 0.0
        except Exception as e:
            logger.warning(f"获取音频时长失败: {e}")
            return 0.0

    def cleanup(self, temp_files):
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {f} - {e}")
        logger.info("临时文件清理完成")


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


def main():
    parser = argparse.ArgumentParser(
        description="多模型支持的视频字幕提取工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("video_path", nargs='?', default="1.mkv", help="输入视频文件路径")
    parser.add_argument("--model-id", default="fireredasr-aed",
                        help="模型标识（格式：{type}-{name}，如whisper-large-v3或fireredasr-aed）")
    parser.add_argument("--model-path", default=None,
                        help="模型路径（用于FireRedASR等需要指定路径的模型）")
    parser.add_argument("--output", "-o", default="output.srt", help="输出字幕文件路径")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--language", "-l", default=None, help="强制语言代码（如zh/en）")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--no-denoise", action="store_true", help="禁用降噪")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")
    parser.add_argument("--max-gpu-memory", type=int, default=3500, help="最大GPU显存(MB)")
    parser.add_argument("--segment-length", type=int, default=30, help="音频分段长度(秒)")
    parser.add_argument("--segment-overlap", type=int, default=5, help="分段重叠长度(秒)")
    parser.add_argument("--memory-threshold", type=float, default=0.85, help="显存使用率阈值(0-1)")
    parser.add_argument("--temp-dir", default="temp_audio", help="临时文件目录")
    parser.add_argument("--batch-size", type=int, default=2, help="批处理大小")
    parser.add_argument("--num-workers", type=int, default=2, help="并行处理数")
    parser.add_argument("--enable-jit", action="store_true", help="启用JIT编译优化")
    parser.add_argument("--enable-cuda-graph", action="store_true", help="启用CUDA图优化")
    parser.add_argument("--use-tensorrt",default=True, action="store_true", help="使用TensorRT加速")
    parser.add_argument("--tensorrt-precision", choices=["fp32", "fp16", "int8"], default="fp16",
                        help="TensorRT精度模式")

    args = parser.parse_args()

    # 设备自动检测
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {args.device}")

    # 检查TensorRT可用性
    if args.use_tensorrt and not TENSORRT_AVAILABLE:
        logger.warning("TensorRT未安装，将使用默认推理模式")
        args.use_tensorrt = False
    else:
        logger.info(f"TensorRT状态: {'启用' if args.use_tensorrt else '禁用'}")
        if args.use_tensorrt:
            logger.info(f"TensorRT精度模式: {args.tensorrt_precision}")

    # 针对FireRedASR的特殊参数处理
    if args.model_id.startswith("fireredasr-"):
        model_type = args.model_id.split("-")[1]
        if not args.model_path:
            args.model_path = os.path.expanduser(f"~/.cache/modelscope/hub/pengzhendong/FireRedASR-{model_type.upper()}-L")
            logger.info(f"使用默认模型路径: {args.model_path}")

    # 构建模型参数
    model_kwargs = {
        "max_gpu_memory": args.max_gpu_memory,
        "max_segment_length": args.segment_length,
        "segment_overlap": args.segment_overlap,
        "memory_cleanup_threshold": args.memory_threshold,
        "temp_dir": args.temp_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "enable_jit": args.enable_jit,
        "enable_cuda_graph": args.enable_cuda_graph,
        "use_tensorrt": args.use_tensorrt,
        "tensorrt_precision": args.tensorrt_precision,
        "compute_type": "int8_float16",
        "download_root": "models"
    }

    if args.model_path:
        model_kwargs["model_path"] = args.model_path

    extractor = VideoSubtitleExtractor(
        model_id=args.model_id,
        device=args.device,
        **model_kwargs
    )

    temp_files = []
    try:
        logger.info("=" * 60)
        logger.info("开始视频字幕提取流程")
        logger.info("=" * 60)

        with Timer("音频提取"):
            logger.info(f"正在从视频文件提取音频: {args.video_path}")
            audio_path = extractor.extract_audio(args.video_path)
            temp_files.append(audio_path)
            logger.info(f"音频提取完成: {audio_path}")

        with Timer("音频预处理"):
            logger.info("开始音频预处理...")
            if not args.no_denoise:
                logger.info("启用音频降噪处理")
            processed_audio = extractor.preprocess_audio(audio_path, denoise=not args.no_denoise)
            temp_files.append(processed_audio)
            logger.info(f"音频预处理完成: {processed_audio}")

        with Timer("语音转录"):
            logger.info("开始语音识别转录...")
            result = extractor.transcribe_audio(processed_audio, args.language, args.temperature)
            if not result["segments"]:
                logger.error("未检测到语音内容，请检查音频文件")
                return
            logger.info(f"转录完成，共识别到 {len(result['segments'])} 个语音片段")

        with Timer("字幕生成"):
            logger.info("开始生成SRT字幕文件...")
            extractor.create_srt_file(result["segments"], args.output)
            logger.info(f"字幕文件生成完成: {args.output}")

        logger.info("=" * 60)
        logger.info("视频字幕提取流程完成！")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"处理失败: {e}")
        traceback.print_exc()
    finally:
        if not args.keep_temp:
            extractor.cleanup(temp_files)
        # 释放模型内存
        extractor.model_wrapper.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已释放GPU显存")


if __name__ == "__main__":
    jieba.initialize()
    main()