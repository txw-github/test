
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
        
    def _load_config(self, config_path: str) -> Dict:
        """加载音频处理配置"""
        default_config = {
            "noise_reduction": {
                "enable": True,
                "strength": 0.5,
                "method": "spectral_gating"
            },
            "voice_enhancement": {
                "enable": True,
                "vocal_isolation": True,
                "frequency_emphasis": [300, 3400]  # 人声频率范围
            },
            "normalization": {
                "enable": True,
                "target_lufs": -16,
                "peak_limit": -1.5
            },
            "chinese_optimization": {
                "enable": True,
                "tone_preservation": True,
                "consonant_enhancement": True
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并配置
                    for key in default_config:
                        if key in user_config:
                            default_config[key].update(user_config[key])
                    return default_config
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}")
        
        return default_config

    def preprocess_audio(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        多级音频预处理以提高中文识别质量
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "enhanced_audio.wav")

        try:
            logger.info(f"🎵 开始高级音频预处理: {audio_path}")

            # 阶段1: 基础格式转换和标准化
            stage1_path = self._stage1_basic_processing(audio_path)
            
            # 阶段2: 降噪和去混响
            stage2_path = self._stage2_noise_reduction(stage1_path)
            
            # 阶段3: 语音增强和中文优化
            stage3_path = self._stage3_voice_enhancement(stage2_path)
            
            # 阶段4: 最终优化
            final_path = self._stage4_final_optimization(stage3_path, output_path)
            
            # 清理临时文件
            self._cleanup_temp_files([stage1_path, stage2_path, stage3_path])
            
            logger.info(f"✅ 音频预处理完成: {output_path}")
            return final_path

        except Exception as e:
            logger.error(f"❌ 音频预处理失败: {e}")
            return audio_path

    def _stage1_basic_processing(self, audio_path: str) -> str:
        """阶段1: 基础处理"""
        temp_path = tempfile.mktemp(suffix="_stage1.wav")
        
        try:
            # 基础格式转换 + 响度标准化
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", str(self.target_sample_rate),
                "-ac", "1",  # 单声道
                "-af", f"loudnorm=I={self.config['normalization']['target_lufs']}:TP={self.config['normalization']['peak_limit']}:LRA=11",
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"基础处理失败: {result.stderr}")
                
            logger.debug("✓ 阶段1: 基础处理完成")
            return temp_path
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _stage2_noise_reduction(self, audio_path: str) -> str:
        """阶段2: 多级降噪"""
        temp_path = tempfile.mktemp(suffix="_stage2.wav")
        
        try:
            if not self.config["noise_reduction"]["enable"]:
                # 如果禁用降噪，直接复制
                import shutil
                shutil.copy2(audio_path, temp_path)
                return temp_path
            
            # 构建降噪滤镜链
            filters = []
            
            # 1. 高通滤波器去除低频噪声
            filters.append("highpass=f=80")
            
            # 2. 低通滤波器去除高频噪声
            filters.append("lowpass=f=8000")
            
            # 3. 动态降噪
            filters.append("afftdn=nf=-25:nt=w:om=o:tn=1")
            
            # 4. 去除爆音和咔嗒声
            filters.append("declick=t=w:l=2")
            
            # 5. 去除嘶嘶声
            filters.append("dehiss=m=o")
            
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
                    "-af", "highpass=f=80,lowpass=f=8000",
                    "-acodec", "pcm_s16le",
                    "-loglevel", "error",
                    temp_path
                ]
                subprocess.run(simple_cmd, check=True, capture_output=True)
                
            logger.debug("✓ 阶段2: 降噪处理完成")
            return temp_path
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _stage3_voice_enhancement(self, audio_path: str) -> str:
        """阶段3: 语音增强和中文优化"""
        temp_path = tempfile.mktemp(suffix="_stage3.wav")
        
        try:
            if not self.config["voice_enhancement"]["enable"]:
                import shutil
                shutil.copy2(audio_path, temp_path)
                return temp_path
            
            filters = []
            
            # 1. 语音频段增强 (300-3400Hz为人声主要频段)
            freq_range = self.config["voice_enhancement"]["frequency_emphasis"]
            filters.append(f"equalizer=f={freq_range[0]}:width_type=h:width=1000:g=3")
            filters.append(f"equalizer=f=1000:width_type=h:width=800:g=2")
            filters.append(f"equalizer=f=2000:width_type=h:width=600:g=2")
            
            # 2. 中文声调优化 - 保护音调变化
            if self.config["chinese_optimization"]["tone_preservation"]:
                filters.append("acompressor=threshold=0.5:ratio=2:attack=5:release=50")
            
            # 3. 辅音增强 - 提高清晰度
            if self.config["chinese_optimization"]["consonant_enhancement"]:
                filters.append("equalizer=f=4000:width_type=h:width=2000:g=1.5")
            
            # 4. 动态范围压缩
            filters.append("compand=attacks=0.3:decays=0.8:points=-80/-80|-45/-30|-27/-20|-12/-8|-6/-6:soft-knee=6")
            
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

    def _stage4_final_optimization(self, audio_path: str, output_path: str) -> str:
        """阶段4: 最终优化"""
        try:
            # 最终处理: 标准化和限制器
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "alimiter=level_in=1:level_out=0.9:limit=0.95:attack=7:release=100",
                "-acodec", "pcm_s16le",
                "-loglevel", "error",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"最终优化失败: {result.stderr}")
                
            logger.debug("✓ 阶段4: 最终优化完成")
            return output_path
            
        except Exception as e:
            # 如果最终优化失败，直接复制
            import shutil
            shutil.copy2(audio_path, output_path)
            return output_path

    def _cleanup_temp_files(self, temp_files: list):
        """清理临时文件"""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
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
                
                # 简单的VAD
                if energy > 0.001 and 20 < zcr < 120:
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
            
            return 20.0  # 默认SNR
        except:
            return 20.0

    def _chinese_speech_analysis(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """中文语音特征分析"""
        try:
            # 分析中文语音的特征频率分布
            freq_analysis = self._analyze_frequency_content(audio_data, sample_rate)
            
            # 中文语音的理想频率分布
            ideal_mid_freq = 0.7  # 中频应占主导
            ideal_high_freq = 0.2  # 适量高频
            ideal_low_freq = 0.1   # 少量低频
            
            mid_score = 1.0 - abs(freq_analysis.get("mid_freq_ratio", 0.5) - ideal_mid_freq)
            high_score = 1.0 - abs(freq_analysis.get("high_freq_ratio", 0.2) - ideal_high_freq)
            low_score = 1.0 - abs(freq_analysis.get("low_freq_ratio", 0.3) - ideal_low_freq)
            
            # 综合评分
            chinese_score = (mid_score * 0.6 + high_score * 0.3 + low_score * 0.1) * 100
            return max(0, min(100, float(chinese_score)))
            
        except:
            return 70.0  # 默认评分

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
            if 0.05 <= rms <= 0.3:
                score += 20
            elif 0.01 <= rms <= 0.5:
                score += 15
            else:
                score += 8
            
            # 噪声水平评分 (25%)
            noise_score = metrics["noise_level"]
            score += noise_score * 0.25
            
            # 语音清晰度评分 (25%)
            clarity = metrics["speech_clarity"]
            score += clarity * 25
            
            # 中文适配度评分 (15%)
            chinese_score = metrics["chinese_speech_score"]
            score += chinese_score * 0.15
            
            return max(0, min(100, float(score)))
            
        except:
            return 50.0

    def _generate_recommendations(self, metrics: Dict) -> list:
        """生成优化建议"""
        recommendations = []
        
        try:
            if metrics["sample_rate"] < 16000:
                recommendations.append("建议提高音频采样率至16kHz以上")
            
            if metrics["rms_level"] < 0.01:
                recommendations.append("音频音量过低，建议增加增益")
            elif metrics["rms_level"] > 0.5:
                recommendations.append("音频音量过高，可能存在削波失真")
            
            if metrics["noise_level"] < 15:
                recommendations.append("检测到较高噪声，建议进行降噪处理")
            
            if metrics["speech_clarity"] < 0.5:
                recommendations.append("语音清晰度较低，建议使用语音增强")
            
            if metrics["chinese_speech_score"] < 60:
                recommendations.append("建议启用中文语音优化设置")
            
            freq_analysis = metrics.get("frequency_analysis", {})
            speech_dominance = freq_analysis.get("speech_freq_dominance", 0.5)
            if speech_dominance < 0.4:
                recommendations.append("语音频段能量不足，建议使用EQ增强")
                
        except:
            pass
        
        return recommendations

# 保持向后兼容
class AudioPreprocessor(AdvancedAudioPreprocessor):
    """向后兼容的音频预处理器"""
    pass
