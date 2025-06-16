
import os
import logging
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, Dict
import tempfile
import subprocess
import json

logger = logging.getLogger(__name__)

class AdvancedAudioPreprocessor:
    """高级音频预处理器 - 针对中文语音识别优化"""

    def __init__(self, target_sample_rate: int = 16000, config_path: str = "audio_config.json"):
        self.target_sample_rate = target_sample_rate
        self.config = self._load_config(config_path)
        
        # 设置FFmpeg路径（如果用户有自定义路径）
        ffmpeg_path = os.environ.get("FFMPEG_PATH")
        if ffmpeg_path:
            os.environ["PATH"] += os.pathsep + ffmpeg_path
        
    def _load_config(self, config_path: str) -> Dict:
        """加载音频处理配置"""
        default_config = {
            "noise_reduction": {
                "enable": True,
                "strength": 0.7,
                "method": "multi_stage",
                "adaptive": True
            },
            "voice_enhancement": {
                "enable": True,
                "vocal_isolation": True,
                "frequency_emphasis": [300, 3400],
                "dynamic_range_compression": True,
                "consonant_enhancement": True
            },
            "normalization": {
                "enable": True,
                "target_lufs": -16,
                "peak_limit": -1.5,
                "loudness_range": 11
            },
            "chinese_optimization": {
                "enable": True,
                "tone_preservation": True,
                "consonant_enhancement": True,
                "frequency_profile": "mandarin"
            },
            "advanced_features": {
                "speech_enhancement": True,
                "background_music_suppression": True,
                "reverb_reduction": True,
                "click_removal": True
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 深度合并配置
                    for key in default_config:
                        if key in user_config and isinstance(user_config[key], dict):
                            default_config[key].update(user_config[key])
                    return default_config
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}")
        
        return default_config

    def preprocess_audio(self, audio_path: str, output_path: Optional[str] = None, quality: str = "high") -> str:
        """
        多级音频预处理以提高中文识别质量
        quality: "high", "balanced", "fast"
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "enhanced_audio.wav")

        try:
            logger.info(f"🎵 开始高级音频预处理: {audio_path} (质量: {quality})")

            # 根据质量等级调整处理参数
            quality_configs = {
                "high": {
                    "noise_strength": 0.8,
                    "voice_enhancement": True,
                    "advanced_processing": True,
                    "stages": 5
                },
                "balanced": {
                    "noise_strength": 0.6,
                    "voice_enhancement": True,
                    "advanced_processing": True,
                    "stages": 4
                },
                "fast": {
                    "noise_strength": 0.4,
                    "voice_enhancement": False,
                    "advanced_processing": False,
                    "stages": 3
                }
            }
            
            config = quality_configs.get(quality, quality_configs["balanced"])

            # 阶段1: 基础格式转换和预处理
            stage1_path = self._stage1_format_conversion(audio_path)
            
            # 阶段2: 高级降噪
            stage2_path = self._stage2_advanced_noise_reduction(stage1_path, config["noise_strength"])
            
            # 阶段3: 语音增强（可选）
            if config["voice_enhancement"]:
                stage3_path = self._stage3_voice_enhancement(stage2_path)
            else:
                stage3_path = stage2_path
            
            # 阶段4: 中文语音优化
            stage4_path = self._stage4_chinese_optimization(stage3_path)
            
            # 阶段5: 高级后处理（可选）
            if config["advanced_processing"]:
                stage5_path = self._stage5_advanced_postprocessing(stage4_path)
            else:
                stage5_path = stage4_path
            
            # 最终阶段: 标准化输出
            final_path = self._stage_final_normalization(stage5_path, output_path)
            
            # 清理临时文件
            temp_files = [stage1_path, stage2_path, stage3_path, stage4_path, stage5_path]
            self._cleanup_temp_files([f for f in temp_files if f != output_path])
            
            logger.info(f"✅ 音频预处理完成: {output_path}")
            return final_path

        except Exception as e:
            logger.error(f"❌ 音频预处理失败: {e}")
            return audio_path

    def _stage1_format_conversion(self, audio_path: str) -> str:
        """阶段1: 格式转换和基础预处理"""
        temp_path = tempfile.mktemp(suffix="_stage1.wav")
        
        try:
            # 基础转换 + 初步标准化
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", str(self.target_sample_rate),
                "-ac", "1",  # 单声道
                "-af", "volume=1.0",  # 保持原始音量
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"格式转换失败: {result.stderr}")
                
            logger.debug("✓ 阶段1: 格式转换完成")
            return temp_path
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _stage2_advanced_noise_reduction(self, audio_path: str, strength: float) -> str:
        """阶段2: 高级多层降噪 - Windows兼容版"""
        temp_path = tempfile.mktemp(suffix="_stage2.wav")
        
        try:
            if not self.config["noise_reduction"]["enable"]:
                import shutil
                shutil.copy2(audio_path, temp_path)
                return temp_path
            
            # 分层处理，避免复杂滤镜链在Windows下的兼容性问题
            # 第一层：基础滤波
            temp1_path = tempfile.mktemp(suffix="_stage2_1.wav")
            cmd1 = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "highpass=f=85,lowpass=f=7500",
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp1_path
            ]
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True)
            if result1.returncode != 0:
                logger.warning(f"基础滤波失败: {result1.stderr}")
                # 直接复制原文件
                import shutil
                shutil.copy2(audio_path, temp_path)
                return temp_path
            
            # 第二层：自适应降噪（简化版）
            cmd2 = [
                "ffmpeg", "-y", "-i", temp1_path,
                "-af", "afftdn=nf=-12",  # 使用更保守的参数
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp_path
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            if result2.returncode != 0:
                logger.warning("自适应降噪失败，使用基础处理结果")
                # 使用第一层的结果
                import shutil
                shutil.copy2(temp1_path, temp_path)
            
            # 清理临时文件
            if os.path.exists(temp1_path):
                os.remove(temp1_path)
                
            logger.debug("✓ 阶段2: 降噪处理完成")
            return temp_path
            
        except Exception as e:
            # 清理所有临时文件
            for cleanup_path in [temp_path, temp1_path if 'temp1_path' in locals() else None]:
                if cleanup_path and os.path.exists(cleanup_path):
                    try:
                        os.remove(cleanup_path)
                    except:
                        pass
            
            # 降噪完全失败时，直接复制原文件
            logger.warning(f"降噪处理失败: {e}")
            import shutil
            shutil.copy2(audio_path, temp_path)
            return temp_path

    def _stage3_voice_enhancement(self, audio_path: str) -> str:
        """阶段3: 语音增强 - Windows兼容版"""
        temp_path = tempfile.mktemp(suffix="_stage3.wav")
        
        try:
            # 使用更简单但兼容性更好的EQ设置
            eq_filters = [
                "equalizer=f=300:width_type=h:width=1000:g=2",
                "equalizer=f=1000:width_type=h:width=800:g=1.5",
                "equalizer=f=3000:width_type=h:width=1000:g=1"
            ]
            
            # 分步骤处理，避免复杂的滤镜链
            temp1_path = tempfile.mktemp(suffix="_stage3_1.wav")
            
            # 第一步：EQ处理
            cmd1 = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", ",".join(eq_filters),
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp1_path
            ]
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True)
            if result1.returncode != 0:
                logger.warning("EQ处理失败，跳过语音增强")
                import shutil
                shutil.copy2(audio_path, temp_path)
                return temp_path
            
            # 第二步：简化的压缩处理
            cmd2 = [
                "ffmpeg", "-y", "-i", temp1_path,
                "-af", "acompressor=threshold=0.4:ratio=2:attack=5:release=50",
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp_path
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            if result2.returncode != 0:
                logger.warning("压缩处理失败，使用EQ处理结果")
                import shutil
                shutil.copy2(temp1_path, temp_path)
            
            # 清理临时文件
            if os.path.exists(temp1_path):
                os.remove(temp1_path)
                
            logger.debug("✓ 阶段3: 语音增强完成")
            return temp_path
            
        except Exception as e:
            # 清理临时文件
            for cleanup_path in [temp_path, temp1_path if 'temp1_path' in locals() else None]:
                if cleanup_path and os.path.exists(cleanup_path):
                    try:
                        os.remove(cleanup_path)
                    except:
                        pass
            
            logger.warning(f"语音增强失败: {e}")
            import shutil
            shutil.copy2(audio_path, temp_path)
            return temp_path

    def _stage4_chinese_optimization(self, audio_path: str) -> str:
        """阶段4: 中文语音专项优化 - 简化版"""
        temp_path = tempfile.mktemp(suffix="_stage4.wav")
        
        try:
            if not self.config["chinese_optimization"]["enable"]:
                import shutil
                shutil.copy2(audio_path, temp_path)
                return temp_path
            
            # 使用更简单的中文优化，避免复杂滤镜
            simple_filters = [
                "equalizer=f=400:width_type=h:width=600:g=1.5",
                "equalizer=f=2000:width_type=h:width=1000:g=1"
            ]
            
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", ",".join(simple_filters),
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"中文优化失败: {result.stderr}")
                # 失败时直接复制原文件
                import shutil
                shutil.copy2(audio_path, temp_path)
                
            logger.debug("✓ 阶段4: 中文优化完成")
            return temp_path
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            logger.warning(f"中文优化处理失败: {e}")
            import shutil
            shutil.copy2(audio_path, temp_path)
            return temp_path

    def _stage5_advanced_postprocessing(self, audio_path: str) -> str:
        """阶段5: 高级后处理 - 简化版"""
        temp_path = tempfile.mktemp(suffix="_stage5.wav")
        
        try:
            # 只保留最兼容的后处理
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=0.02",
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("后处理失败，使用原始音频")
                import shutil
                shutil.copy2(audio_path, temp_path)
                
            logger.debug("✓ 阶段5: 高级后处理完成")
            return temp_path
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            logger.warning(f"高级后处理失败: {e}")
            import shutil
            shutil.copy2(audio_path, temp_path)
            return temp_path

    def _stage_final_normalization(self, audio_path: str, output_path: str) -> str:
        """最终阶段: 标准化输出"""
        try:
            # 最终标准化和限制
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", f"loudnorm=I={self.config['normalization']['target_lufs']}:TP={self.config['normalization']['peak_limit']}:LRA={self.config['normalization']['loudness_range']},alimiter=level_in=1:level_out=0.95:limit=0.98",
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"最终标准化失败: {result.stderr}")
                
            logger.debug("✓ 最终阶段: 标准化完成")
            return output_path
            
        except Exception as e:
            # 最终失败时直接复制
            import shutil
            shutil.copy2(audio_path, output_path)
            return output_path

    def _cleanup_temp_files(self, temp_files: list):
        """清理临时文件"""
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

    def analyze_audio_quality(self, audio_path: str) -> Dict:
        """详细的音频质量分析"""
        try:
            audio_data, sample_rate = sf.read(audio_path)
            
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # 计算各种质量指标
            metrics = {
                "duration": len(audio_data) / sample_rate,
                "sample_rate": sample_rate,
                "rms_level": float(np.sqrt(np.mean(audio_data**2))),
                "peak_level": float(np.max(np.abs(audio_data))),
                "dynamic_range": self._calculate_dynamic_range(audio_data),
                "frequency_analysis": self._analyze_frequency_content(audio_data, sample_rate),
                "speech_clarity": self._estimate_speech_clarity(audio_data, sample_rate),
                "noise_level": self._estimate_noise_level(audio_data),
                "chinese_speech_score": self._chinese_speech_analysis(audio_data, sample_rate)
            }

            # 综合质量评分
            quality_score = self._calculate_comprehensive_score(metrics)
            metrics["overall_score"] = quality_score
            metrics["recommendations"] = self._generate_recommendations(metrics)

            logger.info(f"📊 音频质量分析 - 时长: {metrics['duration']:.1f}s, "
                       f"综合评分: {quality_score:.1f}, "
                       f"中文适配度: {metrics['chinese_speech_score']:.1f}")

            return metrics

        except Exception as e:
            logger.error(f"音频质量分析失败: {e}")
            return {}

    def _calculate_dynamic_range(self, audio_data: np.ndarray) -> float:
        """计算动态范围"""
        try:
            rms_values = []
            frame_size = int(0.1 * self.target_sample_rate)  # 100ms帧
            
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size]
                rms = np.sqrt(np.mean(frame**2))
                if rms > 0:
                    rms_values.append(20 * np.log10(rms))
            
            if len(rms_values) > 1:
                return float(np.max(rms_values) - np.min(rms_values))
            return 0.0
        except:
            return 0.0

    def _analyze_frequency_content(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """频率内容分析"""
        try:
            from scipy import signal
            
            # 计算功率谱密度
            frequencies, psd = signal.welch(audio_data, sample_rate, nperseg=1024)
            
            # 分析不同频段的能量
            low_freq = np.sum(psd[(frequencies >= 80) & (frequencies <= 300)])
            mid_freq = np.sum(psd[(frequencies >= 300) & (frequencies <= 3400)])
            high_freq = np.sum(psd[(frequencies >= 3400) & (frequencies <= 8000)])
            
            total_energy = low_freq + mid_freq + high_freq
            
            if total_energy > 0:
                return {
                    "low_freq_ratio": float(low_freq / total_energy),
                    "mid_freq_ratio": float(mid_freq / total_energy),
                    "high_freq_ratio": float(high_freq / total_energy),
                    "speech_freq_dominance": float(mid_freq / total_energy)
                }
            else:
                return {"low_freq_ratio": 0, "mid_freq_ratio": 0, "high_freq_ratio": 0, "speech_freq_dominance": 0}
                
        except ImportError:
            logger.debug("scipy未安装，跳过频率分析")
            return {"speech_freq_dominance": 0.5}  # 默认值
        except:
            return {"speech_freq_dominance": 0.5}

    def _estimate_speech_clarity(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """估算语音清晰度"""
        try:
            # 计算语音活动检测
            frame_length = int(0.025 * sample_rate)  # 25ms
            hop_length = int(0.01 * sample_rate)     # 10ms
            
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy = np.sum(frame**2)
                zcr = np.sum(np.abs(np.diff(np.sign(frame))))
                
                # 改进的VAD
                if energy > 0.0005 and 15 < zcr < 150:
                    speech_frames += 1
                total_frames += 1
            
            if total_frames > 0:
                return float(speech_frames / total_frames)
            return 0.0
        except:
            return 0.5

    def _estimate_noise_level(self, audio_data: np.ndarray) -> float:
        """估算噪声水平"""
        try:
            # 使用前后1秒作为静音参考
            silence_samples = int(1.0 * self.target_sample_rate)
            
            if len(audio_data) > 2 * silence_samples:
                start_silence = audio_data[:silence_samples]
                end_silence = audio_data[-silence_samples:]
                silence_rms = np.sqrt(np.mean(np.concatenate([start_silence, end_silence])**2))
                
                overall_rms = np.sqrt(np.mean(audio_data**2))
                
                if overall_rms > 0:
                    snr = 20 * np.log10(overall_rms / (silence_rms + 1e-10))
                    return max(0, min(100, float(snr)))
            
            return 25.0  # 默认SNR
        except:
            return 25.0

    def _chinese_speech_analysis(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """中文语音特征分析"""
        try:
            # 分析中文语音的特征频率分布
            freq_analysis = self._analyze_frequency_content(audio_data, sample_rate)
            
            # 中文语音的理想频率分布
            ideal_mid_freq = 0.65  # 中频应占主导
            ideal_high_freq = 0.25  # 适量高频
            ideal_low_freq = 0.1   # 少量低频
            
            mid_score = 1.0 - abs(freq_analysis.get("mid_freq_ratio", 0.5) - ideal_mid_freq)
            high_score = 1.0 - abs(freq_analysis.get("high_freq_ratio", 0.2) - ideal_high_freq)
            low_score = 1.0 - abs(freq_analysis.get("low_freq_ratio", 0.3) - ideal_low_freq)
            
            # 综合评分
            chinese_score = (mid_score * 0.6 + high_score * 0.3 + low_score * 0.1) * 100
            return max(0, min(100, float(chinese_score)))
            
        except:
            return 75.0  # 默认评分

    def _calculate_comprehensive_score(self, metrics: Dict) -> float:
        """计算综合质量评分"""
        try:
            score = 0
            
            # 采样率评分 (15%)
            if metrics["sample_rate"] >= 16000:
                score += 15
            elif metrics["sample_rate"] >= 8000:
                score += 10
            else:
                score += 5
            
            # RMS电平评分 (20%)
            rms = metrics["rms_level"]
            if 0.08 <= rms <= 0.4:
                score += 20
            elif 0.02 <= rms <= 0.6:
                score += 15
            else:
                score += 8
            
            # 噪声水平评分 (25%)
            noise_score = metrics["noise_level"]
            score += min(25, noise_score * 0.25)
            
            # 语音清晰度评分 (25%)
            clarity = metrics["speech_clarity"]
            score += clarity * 25
            
            # 中文适配度评分 (15%)
            chinese_score = metrics["chinese_speech_score"]
            score += chinese_score * 0.15
            
            return max(0, min(100, float(score)))
            
        except:
            return 60.0

    def _generate_recommendations(self, metrics: Dict) -> list:
        """生成优化建议"""
        recommendations = []
        
        try:
            if metrics["sample_rate"] < 16000:
                recommendations.append("建议提高音频采样率至16kHz以上")
            
            if metrics["rms_level"] < 0.02:
                recommendations.append("音频音量过低，建议增加增益")
            elif metrics["rms_level"] > 0.6:
                recommendations.append("音频音量过高，可能存在削波失真")
            
            if metrics["noise_level"] < 20:
                recommendations.append("检测到较高噪声，建议进行高级降噪处理")
            
            if metrics["speech_clarity"] < 0.6:
                recommendations.append("语音清晰度较低，建议使用语音增强")
            
            if metrics["chinese_speech_score"] < 70:
                recommendations.append("建议启用中文语音优化设置")
            
            freq_analysis = metrics.get("frequency_analysis", {})
            speech_dominance = freq_analysis.get("speech_freq_dominance", 0.5)
            if speech_dominance < 0.5:
                recommendations.append("语音频段能量不足，建议使用EQ增强")
                
        except:
            pass
        
        return recommendations

# 保持向后兼容
class AudioPreprocessor(AdvancedAudioPreprocessor):
    """向后兼容的音频预处理器"""
    
    def process_audio(self, audio_path: str, output_path: Optional[str] = None, quality: str = "balanced") -> str:
        """向后兼容的音频处理方法"""
        return self.preprocess_audio(audio_path, output_path, quality)
    
    @staticmethod
    def normalize_audio(audio_path, output_path):
        """标准化音频音量 - 保持向后兼容"""
        preprocessor = AdvancedAudioPreprocessor()
        return preprocessor.preprocess_audio(audio_path, output_path, "fast")
