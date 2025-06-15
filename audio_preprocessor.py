
import os
import logging
import numpy as np
import soundfile as sf
from typing import Optional, Tuple
import tempfile

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """音频预处理器 - 提高语音识别质量"""

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def preprocess_audio(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        预处理音频文件以提高识别质量
        
        Args:
            audio_path: 输入音频文件路径
            output_path: 输出音频文件路径（可选）
            
        Returns:
            处理后的音频文件路径
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        if output_path is None:
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "preprocessed_audio.wav")

        try:
            logger.info(f"开始预处理音频: {audio_path}")

            # 加载音频
            audio_data, sample_rate = sf.read(audio_path)
            logger.debug(f"原始音频 - 采样率: {sample_rate}Hz, 长度: {len(audio_data)} 采样点")

            # 转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.debug("已转换为单声道")

            # 重采样到目标采样率
            if sample_rate != self.target_sample_rate:
                audio_data = self._resample_audio(audio_data, sample_rate, self.target_sample_rate)
                sample_rate = self.target_sample_rate
                logger.debug(f"已重采样到 {self.target_sample_rate}Hz")

            # 音频增强处理
            audio_data = self._enhance_audio(audio_data)

            # 保存处理后的音频
            sf.write(output_path, audio_data, sample_rate)
            logger.info(f"音频预处理完成: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"音频预处理失败: {e}")
            # 返回原始文件路径
            return audio_path

    def _resample_audio(self, audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """重采样音频"""
        try:
            import librosa
            resampled = librosa.resample(audio_data, orig_sr=original_rate, target_sr=target_rate)
            return resampled
        except ImportError:
            logger.warning("librosa未安装，跳过重采样")
            return audio_data
        except Exception as e:
            logger.warning(f"重采样失败: {e}")
            return audio_data

    def _enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """音频增强处理"""
        try:
            # 1. 音量标准化
            audio_data = self._normalize_volume(audio_data)

            # 2. 去除静音段
            audio_data = self._remove_silence(audio_data)

            # 3. 降噪处理
            audio_data = self._reduce_noise(audio_data)

            # 4. 预加重处理
            audio_data = self._preemphasis(audio_data)

            return audio_data

        except Exception as e:
            logger.warning(f"音频增强失败: {e}")
            return audio_data

    def _normalize_volume(self, audio_data: np.ndarray) -> np.ndarray:
        """音量标准化"""
        try:
            # 计算RMS值
            rms = np.sqrt(np.mean(audio_data**2))
            
            if rms > 0:
                # 标准化到-20dB
                target_rms = 0.1  # 约-20dB
                scaling_factor = target_rms / rms
                audio_data = audio_data * scaling_factor
                
                # 限制在[-1, 1]范围内
                audio_data = np.clip(audio_data, -1.0, 1.0)
                
                logger.debug("音量标准化完成")

            return audio_data

        except Exception as e:
            logger.warning(f"音量标准化失败: {e}")
            return audio_data

    def _remove_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """去除静音段"""
        try:
            # 计算音频能量
            frame_length = int(0.025 * self.target_sample_rate)  # 25ms帧
            hop_length = int(0.01 * self.target_sample_rate)     # 10ms跳跃

            energy = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy.append(np.sum(frame**2))

            energy = np.array(energy)
            
            # 找到非静音段
            non_silence = energy > threshold * np.max(energy)
            
            if np.any(non_silence):
                start_idx = np.argmax(non_silence) * hop_length
                end_idx = (len(non_silence) - np.argmax(non_silence[::-1]) - 1) * hop_length + frame_length
                
                # 保留少量静音作为缓冲
                buffer = int(0.1 * self.target_sample_rate)  # 100ms缓冲
                start_idx = max(0, start_idx - buffer)
                end_idx = min(len(audio_data), end_idx + buffer)
                
                audio_data = audio_data[start_idx:end_idx]
                logger.debug("静音去除完成")

            return audio_data

        except Exception as e:
            logger.warning(f"静音去除失败: {e}")
            return audio_data

    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """简单的降噪处理"""
        try:
            # 使用简单的高通滤波器去除低频噪声
            from scipy import signal
            
            # 设计高通滤波器 - 截止频率80Hz
            nyquist = self.target_sample_rate / 2
            cutoff = 80 / nyquist
            
            if cutoff < 1.0:
                b, a = signal.butter(4, cutoff, btype='high')
                audio_data = signal.filtfilt(b, a, audio_data)
                logger.debug("降噪处理完成")

            return audio_data

        except ImportError:
            logger.debug("scipy未安装，跳过降噪处理")
            return audio_data
        except Exception as e:
            logger.warning(f"降噪处理失败: {e}")
            return audio_data

    def _preemphasis(self, audio_data: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """预加重处理 - 增强高频成分"""
        try:
            emphasized = np.append(audio_data[0], audio_data[1:] - coeff * audio_data[:-1])
            logger.debug("预加重处理完成")
            return emphasized

        except Exception as e:
            logger.warning(f"预加重处理失败: {e}")
            return audio_data

    def analyze_audio_quality(self, audio_path: str) -> dict:
        """分析音频质量"""
        try:
            audio_data, sample_rate = sf.read(audio_path)
            
            # 转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            quality_metrics = {
                "duration": len(audio_data) / sample_rate,
                "sample_rate": sample_rate,
                "channels": 1,
                "rms_level": np.sqrt(np.mean(audio_data**2)),
                "peak_level": np.max(np.abs(audio_data)),
                "zero_crossing_rate": self._calculate_zcr(audio_data),
                "signal_to_noise_ratio": self._estimate_snr(audio_data)
            }

            # 质量评估
            quality_score = self._calculate_quality_score(quality_metrics)
            quality_metrics["quality_score"] = quality_score
            quality_metrics["quality_level"] = self._get_quality_level(quality_score)

            logger.info(f"音频质量分析 - 时长: {quality_metrics['duration']:.1f}s, "
                       f"质量: {quality_metrics['quality_level']}, "
                       f"评分: {quality_score:.2f}")

            return quality_metrics

        except Exception as e:
            logger.error(f"音频质量分析失败: {e}")
            return {}

    def _calculate_zcr(self, audio_data: np.ndarray) -> float:
        """计算过零率"""
        try:
            zcr = np.mean(np.abs(np.diff(np.sign(audio_data))))
            return zcr
        except:
            return 0.0

    def _estimate_snr(self, audio_data: np.ndarray) -> float:
        """估算信噪比"""
        try:
            # 简单的SNR估算：信号功率与噪声功率比
            signal_power = np.mean(audio_data**2)
            
            # 假设前100ms和后100ms为噪声
            noise_samples = int(0.1 * self.target_sample_rate)
            if len(audio_data) > 2 * noise_samples:
                noise_start = audio_data[:noise_samples]
                noise_end = audio_data[-noise_samples:]
                noise_power = np.mean(np.concatenate([noise_start, noise_end])**2)
                
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    return max(snr, 0)  # 限制最小值为0
            
            return 20.0  # 默认SNR

        except:
            return 20.0

    def _calculate_quality_score(self, metrics: dict) -> float:
        """计算音频质量评分 (0-100)"""
        try:
            score = 0
            
            # 采样率评分 (30分)
            if metrics["sample_rate"] >= 16000:
                score += 30
            elif metrics["sample_rate"] >= 8000:
                score += 20
            else:
                score += 10

            # RMS电平评分 (25分)
            rms = metrics["rms_level"]
            if 0.05 <= rms <= 0.3:  # 理想范围
                score += 25
            elif 0.01 <= rms <= 0.5:  # 可接受范围
                score += 20
            else:
                score += 10

            # 信噪比评分 (25分)
            snr = metrics["signal_to_noise_ratio"]
            if snr >= 30:
                score += 25
            elif snr >= 20:
                score += 20
            elif snr >= 10:
                score += 15
            else:
                score += 10

            # 时长评分 (20分)
            duration = metrics["duration"]
            if duration >= 1:  # 至少1秒
                score += 20
            elif duration >= 0.5:
                score += 15
            else:
                score += 10

            return min(score, 100)

        except:
            return 50.0  # 默认评分

    def _get_quality_level(self, score: float) -> str:
        """获取质量等级"""
        if score >= 80:
            return "优秀"
        elif score >= 60:
            return "良好"
        elif score >= 40:
            return "一般"
        else:
            return "较差"
