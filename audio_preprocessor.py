
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
        """阶段2: 高级多层降噪"""
        temp_path = tempfile.mktemp(suffix="_stage2.wav")
        
        try:
            if not self.config["noise_reduction"]["enable"]:
                import shutil
                shutil.copy2(audio_path, temp_path)
                return temp_path
            
            # 构建多层降噪滤镜链
            filters = []
            
            # 1. 高通滤波器 - 去除低频噪声
            filters.append("highpass=f=85")
            
            # 2. 低通滤波器 - 去除高频噪声  
            filters.append("lowpass=f=7500")
            
            # 3. 自适应降噪
            filters.append(f"afftdn=nf=-20:nt=w:om=o:tn=1:tf=0.5")
            
            # 4. 去除点击声和爆音
            filters.append("declick=t=w:l=2")
            
            # 5. 去除嘶嘶声
            filters.append("dehiss=m=o")
            
            # 6. 动态降噪 (根据强度调整)
            if strength > 0.6:
                filters.append(f"anlmdn=s={strength}:p=0.002:r=0.002:m=15")
            
            # 7. 门限降噪
            filters.append("agate=threshold=0.1:ratio=2:attack=10:release=100")
            
            filter_chain = ",".join(filters)
            
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", filter_chain,
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # 降噪失败时使用简化版本
                logger.warning("高级降噪失败，使用基础降噪")
                simple_cmd = [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-af", "highpass=f=85,lowpass=f=7500,afftdn=nf=-15",
                    "-acodec", "pcm_s16le",
                    "-loglevel", "error",
                    temp_path
                ]
                subprocess.run(simple_cmd, check=True, capture_output=True)
                
            logger.debug("✓ 阶段2: 高级降噪完成")
            return temp_path
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _stage3_voice_enhancement(self, audio_path: str) -> str:
        """阶段3: 语音增强"""
        temp_path = tempfile.mktemp(suffix="_stage3.wav")
        
        try:
            filters = []
            
            # 1. 语音频段增强 (中文语音优化)
            filters.append("equalizer=f=300:width_type=h:width=1000:g=3")
            filters.append("equalizer=f=800:width_type=h:width=800:g=2.5")
            filters.append("equalizer=f=1600:width_type=h:width=600:g=2")
            filters.append("equalizer=f=3200:width_type=h:width=800:g=1.5")
            
            # 2. 中文声调保护压缩
            filters.append("acompressor=threshold=0.4:ratio=2.5:attack=3:release=40:makeup=1")
            
            # 3. 辅音增强 - 提高清晰度
            filters.append("equalizer=f=4000:width_type=h:width=2000:g=2")
            filters.append("equalizer=f=6000:width_type=h:width=1500:g=1")
            
            # 4. 多频段压缩
            filters.append("mcompand=0.005,0.1 6 -47/-40,-34/-34,-17/-33 100 | 0.003,0.05 6 -47/-40,-34/-34,-17/-33 400 | 0.000625,0.0125 6 -47/-40,-34/-34,-15/-33 1600 | 0.0001,0.025 6 -47/-40,-34/-34,-31/-31,-0/-30 6400 | 0,0.025 6 -38/-31,-28/-28,-0/-25 22000")
            
            filter_chain = ",".join(filters)
            
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", filter_chain,
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"语音增强失败: {result.stderr}")
                
            logger.debug("✓ 阶段3: 语音增强完成")
            return temp_path
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _stage4_chinese_optimization(self, audio_path: str) -> str:
        """阶段4: 中文语音专项优化"""
        temp_path = tempfile.mktemp(suffix="_stage4.wav")
        
        try:
            if not self.config["chinese_optimization"]["enable"]:
                import shutil
                shutil.copy2(audio_path, temp_path)
                return temp_path
            
            filters = []
            
            # 1. 中文音调频率保护
            filters.append("equalizer=f=200:width_type=h:width=400:g=1")  # 基频保护
            filters.append("equalizer=f=400:width_type=h:width=600:g=2")  # 二次谐波
            
            # 2. 中文辅音清晰度增强
            filters.append("equalizer=f=2500:width_type=h:width=1000:g=1.5")
            filters.append("equalizer=f=5000:width_type=h:width=1500:g=1.2")
            
            # 3. 语音活动检测优化压缩
            filters.append("acompressor=threshold=0.3:ratio=3:attack=2:release=30:knee=2")
            
            # 4. 中文语音特有的动态范围优化
            filters.append("dynaudnorm=framelen=500:gausssize=31:peak=0.95:maxgain=10:targetrms=0.20")
            
            filter_chain = ",".join(filters)
            
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", filter_chain,
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"中文优化失败: {result.stderr}")
                
            logger.debug("✓ 阶段4: 中文优化完成")
            return temp_path
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _stage5_advanced_postprocessing(self, audio_path: str) -> str:
        """阶段5: 高级后处理"""
        temp_path = tempfile.mktemp(suffix="_stage5.wav")
        
        try:
            filters = []
            
            # 1. 去混响
            if self.config["advanced_features"]["reverb_reduction"]:
                filters.append("aderivative")
                filters.append("aintegral")
            
            # 2. 背景音乐抑制
            if self.config["advanced_features"]["background_music_suppression"]:
                filters.append("extrastereo=m=0.5")  # 立体声分离
                filters.append("earwax")  # 人声突出
            
            # 3. 最终清理
            filters.append("silenceremove=start_periods=1:start_silence=0.1:start_threshold=0.02")
            
            if filters:
                filter_chain = ",".join(filters)
                
                cmd = [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-af", filter_chain,
                    "-acodec", "pcm_s16le",
                    "-loglevel", "error",
                    temp_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    # 后处理失败时直接复制
                    import shutil
                    shutil.copy2(audio_path, temp_path)
            else:
                import shutil
                shutil.copy2(audio_path, temp_path)
                
            logger.debug("✓ 阶段5: 高级后处理完成")
            return temp_path
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # 失败时直接复制原文件
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
