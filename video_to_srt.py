import os
import time
import torch
import whisper
import argparse
import traceback
from datetime import timedelta
from moviepy.editor import VideoFileClip
from typing import List, Dict, Any, Optional
os.environ["PATH"] += os.pathsep + r"D:\code\ffmpeg\bin"  # 替换为您的 FFmpeg 路径
# $env: PATH += ";D:\code\Git\bin"

class Timer:
    """简单的计时器类"""

    def __init__(self, name="任务"):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        print(f"开始 {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"{self.name} 完成，耗时: {duration:.2f} 秒")


class VideoSubtitleExtractor:
    def __init__(self, model_size="base", device="cuda", download_root=".whisper_models"):
        """初始化字幕提取器"""
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"使用设备: {self.device}")

        if self.device == "cuda":
            # 检查GPU可用性
            if not torch.cuda.is_available():
                print("警告: CUDA不可用，将使用CPU")
                self.device = "cpu"
            else:
                gpu_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                print(f"GPU: {gpu_name} ({total_memory:.2f} GB 显存)")

        # 加载模型
        print(f"加载Whisper模型: {model_size}")
        self.model = whisper.load_model(model_size, device=self.device, download_root=download_root)
        print(f"模型加载完成，参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 确保工作目录存在
        self.ensure_dir_exists(os.getcwd())

    def ensure_dir_exists(self, path: str):
        """确保目录存在，若不存在则创建"""
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
                print(f"创建目录: {path}")
            except Exception as e:
                print(f"无法创建目录 {path}: {e}")
                raise

    def check_file(self, path: str, operation: str):
        """检查文件是否存在并可访问"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        if not os.path.isfile(path):
            raise ValueError(f"不是有效的文件: {path}")

        if operation == "read" and not os.access(path, os.R_OK):
            raise PermissionError(f"没有读取权限: {path}")

        print(f"文件检查通过: {path}")

    def extract_audio(self, video_path: str, audio_path: str = None) -> Optional[str]:
        """从视频中提取音频"""
        if not audio_path:
            audio_path = "extracted_audio.wav"

        print(f"准备从视频提取音频: {video_path}")
        self.check_file(video_path, "read")

        try:
            with Timer("音频提取"):
                video = VideoFileClip(video_path)
                audio = video.audio
                audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)

            self.check_file(audio_path, "read")
            print(f"音频已成功提取至: {audio_path}")
            return audio_path

        except Exception as e:
            print(f"提取音频时出错: {e}")
            traceback.print_exc()  # 打印详细堆栈信息
            return None

    def preprocess_audio(self, audio_path: str) -> str:
        """音频预处理（音量标准化）"""
        output_path = "processed_audio.wav"

        print(f"准备预处理音频: {audio_path}")
        self.check_file(audio_path, "read")

        try:
            with Timer("音频预处理"):
                # 为简化逻辑，直接返回原音频（可根据需要添加更多处理）
                return audio_path

        except Exception as e:
            print(f"音频预处理出错: {e}")
            traceback.print_exc()
            return audio_path  # 返回原始音频作为备用

    def transcribe_audio(self, audio_path: str, language: Optional[str] = None,
                         temperature: float = 0.2) -> Dict[str, Any]:
        """直接转录整个音频（不分块）"""
        print(f"准备转录音频: {audio_path}")
        self.check_file(audio_path, "read")

        try:
            print(f"开始转录音频... (这可能需要一些时间)")

            # 配置转录参数
            options = {
                "language": language,
                "verbose": False,
                "task": "transcribe",
                "temperature": temperature,
                "fp16": self.device == "cuda",
            }

            # 执行转录
            with Timer("音频转录"):
                # 清理CUDA缓存
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                # 直接处理整个音频文件
                result = self.model.transcribe(audio_path, **options)

                # 释放显存
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                print(f"音频转录完成，识别到 {len(result['segments'])} 个语音片段")
                return result

        except Exception as e:
            print(f"转录音频时出错: {e}")
            traceback.print_exc()
            return {"segments": []}

    def create_srt_file(self, segments: List[Dict[str, Any]], output_path: str = "output.srt") -> str:
        """创建SRT字幕文件"""
        print(f"正在生成SRT字幕文件...")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    text = segment["text"].strip()

                    # 自动换行（每行不超过40个字符）
                    if len(text) > 40:
                        words = text.split()
                        lines = []
                        current_line = ""

                        for word in words:
                            if len(current_line) + len(word) + 1 > 40:
                                lines.append(current_line)
                                current_line = word
                            else:
                                current_line += " " + word if current_line else word

                        if current_line:
                            lines.append(current_line)

                        text = "\n".join(lines)

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

            self.check_file(output_path, "read")
            print(f"SRT文件已成功保存至: {output_path}")
            return output_path

        except Exception as e:
            print(f"生成SRT文件时出错: {e}")
            traceback.print_exc()
            return None

    def _format_time(self, seconds: float) -> str:
        """将时间格式化为SRT格式: HH:MM:SS,mmm"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

    def cleanup(self):
        """清理临时文件"""
        try:
            files_to_delete = ["extracted_audio.wav", "processed_audio.wav"]

            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    print(f"删除临时文件: {file_path}")
                    os.remove(file_path)

            print("已成功清理临时文件")

        except Exception as e:
            print(f"清理临时文件时出错: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="视频字幕提取工具")
    parser.add_argument("video_path", nargs='?', default="1.mkv", help="输入视频文件路径")
    parser.add_argument("--output", "-o", default="output.srt", help="输出字幕文件路径")
    parser.add_argument("--model", "-m", default="large",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper模型大小")
    parser.add_argument("--device", "-d", default="cuda", choices=["auto", "cuda", "cpu"],
                        help="运行设备")
    parser.add_argument("--language", "-l", default=None, help="语言 (中文, 英语, 日语等)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="转录温度 (0.0-1.0, 较低值更稳定)")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")

    args = parser.parse_args()

    # 检查输入视频文件
    if not os.path.exists(args.video_path):
        print(f"错误：输入视频文件不存在: {args.video_path}")
        return

    print(f"开始处理视频: {args.video_path}")

    # 创建提取器实例
    extractor = VideoSubtitleExtractor(
        model_size=args.model,
        device=args.device
    )

    try:
        # 提取音频
        audio_path = extractor.extract_audio(args.video_path)
        if not audio_path:
            print("音频提取失败，程序终止")
            return

        # 音频预处理
        processed_audio = extractor.preprocess_audio(audio_path)

        # 转录音频（不分块）
        result = extractor.transcribe_audio(
            processed_audio,
            language=args.language,
            temperature=args.temperature
        )

        if not result or not result["segments"]:
            print("未识别到任何语音内容")
            return

        # 创建字幕文件
        srt_path = extractor.create_srt_file(result["segments"], args.output)
        if srt_path:
            print(f"字幕提取完成！文件已保存至: {srt_path}")
        else:
            print("字幕文件生成失败")

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        traceback.print_exc()  # 打印完整堆栈信息
    finally:
        # 清理临时文件
        if not args.keep_temp:
            extractor.cleanup()


if __name__ == "__main__":
    main()