import os
import time

import psutil
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
from faster_whisper import WhisperModel
import gc
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# 配置日志
import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# 尝试导入TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
    logger.info("TensorRT导入成功")
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT未安装，将使用默认推理模式")

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
        
        # 确保CUDA上下文正确初始化
        import pycuda.autoinit
        import pycuda.driver as cuda
        cuda.init()
        
        self._build_or_load()

    def _validate_engine_file(self, engine_file: str) -> bool:
        """验证TensorRT引擎文件是否有效"""
        try:
            if not os.path.exists(engine_file):
                return False
                
            # 检查文件大小
            if os.path.getsize(engine_file) < 1024:  # 小于1KB的文件可能是损坏的
                logger.warning(f"TensorRT引擎文件太小，可能已损坏: {engine_file}")
                return False
                
            # 尝试加载引擎
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            with open(engine_file, 'rb') as f:
                engine_data = f.read()
                if not engine_data:
                    logger.error("引擎文件为空")
                    return False
                    
                # 尝试反序列化
                engine = runtime.deserialize_cuda_engine(engine_data)
                if not engine:
                    logger.error("引擎反序列化失败")
                    return False
                    
                # 检查输入输出
                if engine.num_bindings == 0:
                    logger.error("引擎没有绑定")
                    return False
                    
                # 检查是否有输入张量
                has_input = False
                for i in range(engine.num_bindings):
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT:
                        has_input = True
                        break
                        
                if not has_input:
                    logger.error("引擎没有输入张量")
                    return False
                    
                # 检查内存使用
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                if memory_allocated == 0:
                    logger.warning("引擎加载后没有分配GPU内存")
                    return False
                    
                logger.info(f"引擎验证通过，当前GPU内存使用: {memory_allocated:.2f}MB")
                return True
                
        except Exception as e:
            logger.error(f"引擎文件验证失败: {e}")
            return False

    def _build_or_load(self):
        """构建或加载TensorRT引擎"""
        engine_file = os.path.join(self.model_path, "model.trt")
        
        # 检查引擎文件是否存在且有效
        if os.path.exists(engine_file) and self._validate_engine_file(engine_file):
            logger.info("找到有效的TensorRT引擎文件，开始加载...")
            self._load_engine(engine_file)
        else:
            logger.info("未找到有效的TensorRT引擎文件，开始创建...")
            self._build_engine(engine_file)
            
            # 验证新创建的引擎文件
            if not self._validate_engine_file(engine_file):
                raise RuntimeError("TensorRT引擎创建失败：引擎文件验证未通过")

    def _build_engine(self, engine_file):
        """构建TensorRT引擎"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"找不到模型文件: {self.model_path}")
                
            # 创建ONNX模型
            onnx_path = os.path.join(self.model_path, "model.onnx")
            if not os.path.exists(onnx_path):
                logger.info("未找到ONNX模型，开始创建...")
                try:
                    # 确保模型已加载
                    if not self.model:
                        raise RuntimeError("模型未加载")
                        
                    # 创建模型包装器
                    class ModelWrapper(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                            
                        def forward(self, x):
                            try:
                                # 确保输入维度正确
                                if len(x.shape) != 3:
                                    raise ValueError(f"输入维度错误: {x.shape}，期望 (batch, time, feature)")
                                    
                                # 将输入转换为音频文件
                                temp_path = os.path.join(os.path.dirname(onnx_path), "temp_input.wav")
                                audio_data = x.cpu().numpy()
                                sf.write(temp_path, audio_data, 16000)
                                
                                # 调用transcribe方法
                                results = self.model.transcribe(
                                    batch_uttid=["single"],
                                    batch_wav_path=[temp_path],
                                    args={
                                        "gpu": 1 if x.is_cuda else 0,
                                        "compute_type": "float16" if x.dtype == torch.float16 else "float32"
                                    }
                                )
                                
                                # 清理临时文件
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                                    
                                # 提取文本结果并转换为张量
                                if results and isinstance(results, list) and len(results) > 0:
                                    text = results[0]["text"]
                                    # 确保输出是固定长度的张量
                                    output = torch.zeros(100, device=x.device)  # 固定长度为100
                                    text_tensor = torch.tensor([ord(c) for c in text], device=x.device)
                                    output[:len(text_tensor)] = text_tensor
                                    return output
                                else:
                                    return torch.zeros(100, device=x.device)
                                    
                            except Exception as e:
                                logger.error(f"Forward pass failed: {e}")
                                return torch.zeros(100, device=x.device)
                                
                    # 包装模型
                    wrapped_model = ModelWrapper(self.model)
                    wrapped_model.eval()
                    
                    # 创建示例输入
                    dummy_input = torch.randn(1, 1000, 80, device=self.device)
                    
                    # 导出ONNX模型
                    logger.info("开始导出ONNX模型...")
                    torch.onnx.export(
                        wrapped_model,
                        dummy_input,
                        onnx_path,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={
                            'input': {0: 'batch_size', 1: 'sequence'},
                            'output': {0: 'batch_size'}
                        },
                        opset_version=12,
                        do_constant_folding=True,
                        verbose=True
                    )
                    
                    # 验证ONNX模型
                    import onnx
                    onnx_model = onnx.load(onnx_path)
                    onnx.checker.check_model(onnx_model)
                    logger.info("ONNX模型创建成功并验证通过")
                    
                except Exception as e:
                    logger.error(f"ONNX模型创建失败: {e}")
                    if os.path.exists(onnx_path):
                        try:
                            os.remove(onnx_path)
                        except:
                            pass
                    raise
                    
            # 创建TensorRT引擎
            logger.info("开始创建TensorRT引擎...")
            
            # 确保CUDA上下文正确初始化
            import pycuda.autoinit
            import pycuda.driver as cuda
            cuda.init()
            
            # 创建TensorRT日志记录器
            logger_trt = trt.Logger(trt.Logger.INFO)
            
            # 创建构建器
            builder = trt.Builder(logger_trt)
            if not builder:
                raise RuntimeError("TensorRT构建器创建失败")
                
            # 创建网络
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            if not network:
                raise RuntimeError("TensorRT网络创建失败")
                
            # 创建配置
            config = builder.create_builder_config()
            if not config:
                raise RuntimeError("TensorRT配置创建失败")
                
            # 设置精度
            if self.precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("启用FP16精度")
            elif self.precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("启用INT8精度")
                
            # 设置最大工作空间
            config.max_workspace_size = 1 << 30  # 1GB
            
            # 解析ONNX模型
            logger.info("开始解析ONNX模型...")
            parser = trt.OnnxParser(network, logger_trt)
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX解析错误: {parser.get_error(error)}")
                    raise RuntimeError("ONNX模型解析失败")
                    
            # 设置动态形状
            profile = builder.create_optimization_profile()
            profile.set_shape(
                "input",
                (1, 100, 80),    # 最小形状
                (1, 1000, 80),   # 最优形状
                (1, 2000, 80)    # 最大形状
            )
            config.add_optimization_profile(profile)
            
            # 构建引擎
            logger.info("开始构建TensorRT引擎...")
            serialized_engine = builder.build_serialized_network(network, config)
            if not serialized_engine:
                raise RuntimeError("TensorRT引擎构建失败")
                
            # 保存引擎
            logger.info("保存TensorRT引擎...")
            with open(engine_file, 'wb') as f:
                f.write(serialized_engine)
                
            # 验证保存的引擎文件
            if not self._validate_engine_file(engine_file):
                raise RuntimeError("TensorRT引擎保存失败：文件验证未通过")
                
            logger.info("TensorRT引擎创建成功")
            
        except Exception as e:
            logger.error(f"TensorRT引擎构建失败: {e}")
            # 清理可能损坏的文件
            if os.path.exists(engine_file):
                try:
                    os.remove(engine_file)
                except:
                    pass
            raise

    def _load_engine(self, engine_file):
        """加载TensorRT引擎"""
        try:
            # 确保CUDA上下文正确初始化
            import pycuda.autoinit
            import pycuda.driver as cuda
            cuda.init()
            
            # 创建运行时和引擎
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            with open(engine_file, 'rb') as f:
                engine_data = f.read()
                if not engine_data:
                    raise RuntimeError("TensorRT引擎文件为空")
                self.engine = runtime.deserialize_cuda_engine(engine_data)
                
            if not self.engine:
                raise RuntimeError("TensorRT引擎反序列化失败")
                
            # 创建执行上下文
            self.context = self.engine.create_execution_context()
            if not self.context:
                raise RuntimeError("TensorRT执行上下文创建失败")
                
            # 创建CUDA流
            self.stream = cuda.Stream()
            
            # 初始化输入输出绑定
            self.inputs = []
            self.outputs = []
            self.bindings = []
            
            # 获取引擎的输入输出信息
            for binding in range(self.engine.num_bindings):
                # 获取张量名称
                tensor_name = self.engine.get_tensor_name(binding)
                
                # 获取张量形状
                shape = self.engine.get_tensor_shape(tensor_name)
                if not shape:
                    raise RuntimeError(f"无法获取张量 {tensor_name} 的形状")
                    
                # 获取张量数据类型
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                
                # 计算所需内存大小
                size = trt.volume(shape)
                if size == 0:
                    raise RuntimeError(f"张量 {tensor_name} 的大小为0")
                    
                # 分配主机和设备内存
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                # 添加到绑定列表
                self.bindings.append(int(device_mem))
                
                # 根据张量类型添加到输入或输出列表
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.inputs.append({
                        'host': host_mem,
                        'device': device_mem,
                        'name': tensor_name,
                        'shape': shape,
                        'dtype': dtype
                    })
                else:
                    self.outputs.append({
                        'host': host_mem,
                        'device': device_mem,
                        'name': tensor_name,
                        'shape': shape,
                        'dtype': dtype
                    })
                    
            # 验证输入输出绑定
            if not self.inputs:
                raise RuntimeError("没有找到输入张量")
            if not self.outputs:
                raise RuntimeError("没有找到输出张量")
                
            # 设置动态形状
            if self.context:
                for input_tensor in self.inputs:
                    self.context.set_binding_shape(
                        self.engine.get_binding_index(input_tensor['name']),
                        input_tensor['shape']
                    )
                    
            logger.info(f"TensorRT引擎加载成功，输入: {len(self.inputs)}，输出: {len(self.outputs)}")
            
        except Exception as e:
            logger.error(f"TensorRT引擎加载失败: {e}")
            # 清理资源
            self._cleanup()
            raise
            
    def _cleanup(self):
        """清理TensorRT资源"""
        try:
            # 清理输入输出内存
            for tensor in self.inputs + self.outputs:
                if 'device' in tensor:
                    tensor['device'].free()
                    
            # 清理上下文和引擎
            if self.context:
                self.context = None
            if self.engine:
                self.engine = None
                
            # 重置列表
            self.inputs = []
            self.outputs = []
            self.bindings = []
            
        except Exception as e:
            logger.error(f"TensorRT资源清理失败: {e}")
            
    def __del__(self):
        """析构函数"""
        self._cleanup()

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """执行推理"""
        if not self.inputs or not self.outputs:
            raise RuntimeError("TensorRT引擎未正确初始化")

        try:
            # 检查输入数据形状
            input_shape = self.engine.get_tensor_shape(self.inputs[0]['name'])
            if input_data.shape != tuple(input_shape):
                raise ValueError(f"输入数据形状不匹配: 期望 {input_shape}, 得到 {input_data.shape}")

            # 复制输入数据
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # 执行推理
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # 获取输出
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            
            # 重塑输出数据
            output_shape = self.engine.get_tensor_shape(self.outputs[0]['name'])
            return self.outputs[0]['host'].reshape(output_shape)
            
        except Exception as e:
            logger.error(f"TensorRT推理失败: {e}")
            raise

    def get_gpu_memory_usage(self) -> float:
        """获取当前GPU显存使用量（MB）"""
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.memory_allocated() / (1024 ** 2)
        except Exception as e:
            logger.warning(f"获取GPU显存使用量失败: {e}")
            return 0.0

    def _wrap_model(self, model):
        """包装模型以支持TensorRT"""
        import torch.nn as nn
        
        class Wrapper(nn.Module):
            def __init__(self, m, model_path):
                super().__init__()
                self.m = m
                # 创建临时目录
                self.temp_dir = os.path.join(model_path, "temp_audio")
                os.makedirs(self.temp_dir, exist_ok=True)
                
            def forward(self, x):
                try:
                    # 将输入转换为音频文件
                    temp_path = os.path.join(self.temp_dir, "temp_input.wav")
                    audio_data = x.cpu().numpy()
                    sf.write(temp_path, audio_data, 16000)
                    
                    # 调用transcribe方法
                    results = self.m.transcribe(
                        batch_uttid=["single"],
                        batch_wav_path=[temp_path],
                        args={
                            "gpu": 1 if x.is_cuda else 0,
                            "compute_type": "float16" if x.dtype == torch.float16 else "float32"
                        }
                    )
                    
                    # 清理临时文件
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                    # 提取文本结果并转换为张量
                    if results and isinstance(results, list) and len(results) > 0:
                        text = results[0]["text"]
                        return torch.tensor([ord(c) for c in text], device=x.device)
                    else:
                        return torch.zeros(1, device=x.device)
                        
                except Exception as e:
                    logger.error(f"Forward pass failed: {e}")
                    return torch.zeros(1, device=x.device)
                    
            def __del__(self):
                # 清理临时目录
                try:
                    if os.path.exists(self.temp_dir):
                        for file in os.listdir(self.temp_dir):
                            try:
                                os.remove(os.path.join(self.temp_dir, file))
                            except:
                                pass
                        os.rmdir(self.temp_dir)
                except:
                    pass
                    
        return Wrapper(model, self.model_path)


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
        self.use_half_precision = True  # 默认使用半精度
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
                
                # 检查显存使用情况
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                if memory_allocated > self.max_gpu_memory:
                    logger.warning(f"显存使用超过限制 ({memory_allocated:.2f}MB > {self.max_gpu_memory}MB)，尝试进一步优化...")
                    # 尝试使用更激进的优化
                    if hasattr(self.model, 'model'):
                        if hasattr(self.model.model, 'encoder'):
                            self.model.model.encoder.use_checkpoint = True
                            self.model.model.encoder.parallel = True
                        if hasattr(self.model.model, 'decoder'):
                            self.model.model.decoder.use_checkpoint = True
                            self.model.model.decoder.parallel = True
                    
                    # 再次清理
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                
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
            
            if self.use_tensorrt and TENSORRT_AVAILABLE:
                # 使用TensorRT
                engine_path = os.path.join(self.model_path, "model.trt")
                if not os.path.exists(engine_path):
                    logger.info("未找到TensorRT引擎文件，开始创建...")
                    try:
                        # 先加载PyTorch模型用于创建TensorRT引擎
                        self.model = FireRedAsr.from_pretrained(self.model_type)
                        
                        # 为模型添加forward方法
                        def forward(self, x):
                            """自定义forward方法"""
                            try:
                                # 将输入转换为音频文件
                                temp_path = os.path.join(self.model_path, "temp_input.wav")
                                audio_data = x.cpu().numpy()
                                sf.write(temp_path, audio_data, 16000)
                                
                                # 调用transcribe方法
                                results = self.transcribe(
                                    batch_uttid=["single"],
                                    batch_wav_path=[temp_path],
                                    args={
                                        "gpu": 1 if x.is_cuda else 0,
                                        "compute_type": "float16" if x.dtype == torch.float16 else "float32"
                                    }
                                )
                                
                                # 清理临时文件
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                                
                                # 提取文本结果并转换为张量
                                if results and isinstance(results, list) and len(results) > 0:
                                    text = results[0]["text"]
                                    return torch.tensor([ord(c) for c in text], device=x.device)
                                else:
                                    return torch.zeros(1, device=x.device)
                                    
                            except Exception as e:
                                logger.error(f"Forward pass failed: {e}")
                                return torch.zeros(1, device=x.device)
                        
                        # 将forward方法添加到模型类
                        FireRedAsr.forward = forward
                        
                        # 创建TensorRT引擎
                        self.tensorrt_engine = TensorRTEngine(
                            self.model_path,
                            self.device,
                            precision=self.kwargs.get("tensorrt_precision", "fp16"),
                            model=self.model  # 传入模型实例
                        )
                        logger.info("TensorRT引擎创建成功")
                    except Exception as e:
                        logger.error(f"TensorRT引擎创建失败: {e}")
                        logger.warning("将使用PyTorch模型")
                        self.use_tensorrt = False
                else:
                    try:
                        # 直接加载TensorRT引擎
                        self.tensorrt_engine = TensorRTEngine(
                            self.model_path,
                            self.device,
                            precision=self.kwargs.get("tensorrt_precision", "fp16")
                        )
                        logger.info("TensorRT引擎加载成功")
                    except Exception as e:
                        logger.error(f"TensorRT引擎加载失败: {e}")
                        logger.warning("将使用PyTorch模型")
                        self.use_tensorrt = False
            
            # 如果没有使用TensorRT或TensorRT加载失败，使用PyTorch模型
            if not self.use_tensorrt:
                logger.info("使用PyTorch模型")
                self.model = FireRedAsr.from_pretrained(self.model_type)
                
                if self.device == "cuda" and torch.cuda.is_available():
                    self._force_memory_cleanup()
                    
                    # 使用混合精度训练
                    if self.use_half_precision:
                        # 将模型参数转换为半精度
                        if hasattr(self.model, 'model'):
                            for param in self.model.model.parameters():
                                param.data = param.data.half()
                            # 将模型移动到GPU
                            self.model.model = self.model.model.to(self.device)
                        else:
                            # 如果没有model属性，直接移动到GPU
                            self.model = self.model.to(self.device)
                    else:
                        # 不使用半精度，直接移动到GPU
                        self.model = self.model.to(self.device)
                    
                    if self.enable_memory_optimization:
                        # 启用梯度检查点
                        if hasattr(self.model, 'batch_size'):
                            self.model.batch_size = self.batch_size
                        
                        if hasattr(self.model.model, 'encoder'):
                            self.model.model.encoder.use_checkpoint = True
                            self.model.model.encoder.parallel = True
                        
                        # 启用CUDA图优化
                        if self.kwargs.get("enable_cuda_graph", False):
                            try:
                                self.model = torch.cuda.make_graphed_callables(
                                    self.model,
                                    (torch.randn(1, 1000, 80, device=self.device),)
                                )
                            except Exception as e:
                                logger.warning(f"CUDA图优化失败: {e}")
                        
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
            if self.use_tensorrt and self.tensorrt_engine is not None:
                # 使用TensorRT进行推理
                for audio_file in audio_files:
                    # 加载音频数据
                    audio_data = self._load_audio_data(audio_file)
                    # 执行推理
                    output = self.tensorrt_engine.infer(audio_data)
                    # 处理结果
                    results.append({
                        "segments": [{
                            "start": 0.0,
                            "end": self.get_audio_duration(audio_file),
                            "text": self._decode_output(output)
                        }],
                        "language": "zh"
                    })
            else:
                # 使用PyTorch模型进行推理
                batch_uttids = [f"utt_{i}" for i in range(len(audio_files))]
                batch_results = self.model.transcribe(
                    batch_uttid=batch_uttids,
                    batch_wav_path=audio_files,  # 使用正确的参数名
                    args=decode_params or {}
                )
                
                for i, result in enumerate(batch_results):
                    duration = self.get_audio_duration(audio_files[i])
                    results.append({
                        "segments": [{
                            "start": 0.0,
                            "end": duration,
                            "text": result["text"]
                        }],
                        "language": "zh"
                    })
        
        except Exception as e:
            logger.error(f"批处理转录失败: {e}")
            traceback.print_exc()
        
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

    def _build_decode_params(self, kwargs):
        """构建解码参数"""
        # 基础参数
        base_params = {
            "gpu": 1 if self.device == "cuda" else 0,
            "beam_size": 10,
            "nbest": 1,
            "compute_type": "float16" if self.device == "cuda" else "float32",
            "max_batch_size": 1,
            "chunk_size": 16000
        }
        
        # 针对FireRedASR模型的特殊参数处理
        if self.model_id.startswith("fireredasr-"):
            model_type = self.model_id.split("-")[1]
            if model_type == "aed":
                base_params.update({
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.0,
                    "aed_length_penalty": 0.0,
                    "eos_penalty": 1.0
                })
            elif model_type == "llm":
                base_params.update({
                    "decode_max_len": 0,
                    "decode_min_len": 0,
                    "repetition_penalty": 1.0,
                    "llm_length_penalty": 0.0
                })
        
        # 从kwargs中获取decode_params
        decode_params = kwargs.get("decode_params", {})
        
        # 只保留模型支持的参数
        supported_params = {
            "gpu", "beam_size", "nbest", "compute_type", 
            "max_batch_size", "chunk_size", "decode_max_len",
            "decode_min_len", "repetition_penalty", "llm_length_penalty",
            "softmax_smoothing", "aed_length_penalty", "eos_penalty"
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

            return self._parse_results(results, audio_path)
        except Exception as e:
            logger.error(f"单块转录失败: {e}")
            return {"segments": [], "language": "zh", "duration": duration}

    def _parse_results(self, results: List[Dict], audio_path: str) -> Dict[str, Any]:
        """解析转录结果"""
        if not results:
            return {"segments": [], "language": "zh"}

        # 获取第一个结果（单个音频）
        result = results[0]
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

    def forward(self, x):
        """实现forward方法以支持ONNX导出"""
        try:
            # 将输入转换为音频文件
            temp_path = os.path.join(self.temp_dir, "temp_input.wav")
            audio_data = x.cpu().numpy()
            sf.write(temp_path, audio_data, 16000)
            
            # 调用transcribe方法
            results = self.model.transcribe(
                batch_uttid=["single"],
                batch_wav_path=[temp_path],
                args={
                    "gpu": 1 if x.is_cuda else 0,
                    "compute_type": "float16" if x.dtype == torch.float16 else "float32"
                }
            )
            
            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass
            
            # 提取文本结果并转换为张量
            if results and isinstance(results, list) and len(results) > 0:
                text = results[0]["text"]
                return torch.tensor([ord(c) for c in text], device=x.device)
            else:
                return torch.zeros(1, device=x.device)
                
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            return torch.zeros(1, device=x.device)


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
    """音频预处理类"""
    
    @staticmethod
    def normalize_audio(audio_path, output_path):
        """标准化音频音量"""
        try:
            # 检查输入文件是否存在
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"找不到输入音频文件: {audio_path}")
                
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # 使用临时文件进行标准化
            temp_path = f"{os.path.splitext(output_path)[0]}_temp.wav"
            command = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-ar", "16000", "-ac", "1",
                "-acodec", "pcm_s16le", "-loglevel", "error",
                temp_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"音频标准化失败: {result.stderr}")
                
            # 检查临时文件是否生成成功
            if not os.path.exists(temp_path):
                raise RuntimeError("音频标准化失败：临时文件未生成")
                
            # 重命名临时文件
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError(f"文件重命名失败: {e}")
                
            logger.info("FFmpeg轻量音量标准化完成")
            
        except Exception as e:
            logger.error(f"音频标准化失败: {e}")
            # 清理临时文件
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    @staticmethod
    def denoise_audio(audio_path, output_path):
        """降噪处理"""
        temp_path = None
        try:
            # 检查输入文件是否存在
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"找不到输入音频文件: {audio_path}")
                
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # 使用临时文件进行降噪
            temp_path = f"{os.path.splitext(output_path)[0]}_temp.wav"
            
            # 检查RNN模型文件是否存在
            rnn_model_path = "rnnoise_models/somnolent-hogwash.rnnn"
            if not os.path.exists(rnn_model_path):
                logger.warning(f"RNN降噪模型文件不存在: {rnn_model_path}，跳过降噪处理")
                # 如果模型文件不存在，直接复制输入文件到输出路径
                import shutil
                shutil.copy2(audio_path, output_path)
                return output_path
                
            command = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", f"arnndn=m={rnn_model_path}",
                "-ar", "16000", "-ac", "1",
                "-acodec", "pcm_s16le", "-loglevel", "error",
                temp_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"降噪处理失败: {result.stderr}")
                
            # 检查临时文件是否生成成功
            if not os.path.exists(temp_path):
                raise RuntimeError("降噪处理失败：临时文件未生成")
                
            # 重命名临时文件
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
                logger.info("音频降噪处理完成")
                return output_path
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError(f"文件重命名失败: {e}")
                
        except Exception as e:
            logger.error(f"降噪处理失败: {e}")
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            # 如果降噪失败，返回原始音频文件
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
        """从视频中提取音频"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")
            
        if not os.path.isfile(video_path):
            raise ValueError(f"指定的路径不是文件: {video_path}")
            
        # 检查文件格式
        valid_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv']
        if not any(video_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"不支持的文件格式: {video_path}，支持的格式: {', '.join(valid_extensions)}")
            
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = f"{base_name}_extracted_audio.wav"
        
        try:
            command = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ac", "1", "-ar", "16000",  # 降低采样率到16kHz
                "-acodec", "pcm_s16le", "-loglevel", "error",
                audio_path
            ]
            
            # 检查ffmpeg是否可用
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                raise RuntimeError("FFmpeg未安装或无法访问，请确保FFmpeg已正确安装并添加到系统PATH中")
            
            # 执行音频提取
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"音频提取失败: {result.stderr}")
                
            if not os.path.exists(audio_path):
                raise RuntimeError("音频文件生成失败")
                
            logger.info(f"音频提取完成: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"音频提取失败: {str(e)}")
            # 清理可能生成的临时文件
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            raise

    def preprocess_audio(self, audio_path, denoise=True):
        """预处理音频文件"""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"找不到音频文件: {audio_path}")
                
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            processed_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_processed.wav")
            
            # 音量标准化
            try:
                normalized_path = AudioPreprocessor.normalize_audio(audio_path, processed_path)
                if not normalized_path or not os.path.exists(normalized_path):
                    logger.warning("音频标准化失败，使用原始音频")
                    normalized_path = audio_path
            except Exception as e:
                logger.warning(f"音频标准化失败: {e}，使用原始音频")
                normalized_path = audio_path
                
            # 如果需要降噪
            if denoise:
                try:
                    denoised_path = AudioPreprocessor.denoise_audio(normalized_path, processed_path)
                    if not denoised_path or not os.path.exists(denoised_path):
                        logger.warning("降噪处理失败，使用标准化后的音频")
                        return normalized_path
                    return denoised_path
                except Exception as e:
                    logger.warning(f"降噪处理失败: {e}，使用标准化后的音频")
                    return normalized_path
                    
            return normalized_path
            
        except Exception as e:
            logger.error(f"音频预处理失败: {e}")
            # 如果预处理失败，返回原始音频
            return audio_path

    def transcribe_audio(self, audio_path, language=None, temperature=0.2):
        """音频转录主逻辑"""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"找不到音频文件: {audio_path}")
                
            # 构建解码参数
            decode_params = self._build_decode_params({
                "language": language,
                "temperature": temperature
            })
            
            # 获取音频时长
            audio_duration = self.get_audio_duration(audio_path)
            if audio_duration <= 0:
                raise ValueError(f"无效的音频时长: {audio_duration}")
                
            # 根据音频长度决定是否分块处理
            if audio_duration > 300:  # 超过5分钟自动分块
                return self._chunked_transcribe(audio_path, decode_params, audio_duration)
            
            return self._single_chunk_transcribe(audio_path, decode_params, audio_duration)
            
        except Exception as e:
            logger.error(f"转录失败: {e}")
            return {"segments": [], "language": "zh"}

    def _single_chunk_transcribe(self, audio_path, decode_params, duration):
        """单块转录"""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"找不到音频文件: {audio_path}")
                
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

            return self._parse_results(results, audio_path)
            
        except Exception as e:
            logger.error(f"单块转录失败: {e}")
            return {"segments": [], "language": "zh", "duration": duration}

    def cleanup(self, temp_files):
        """清理临时文件"""
        if not temp_files:
            return
            
        for f in temp_files:
            if f and os.path.exists(f):
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
        # 释放模型内存
        extractor.model_wrapper.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已释放GPU显存")


if __name__ == "__main__":
    jieba.initialize()
    main()