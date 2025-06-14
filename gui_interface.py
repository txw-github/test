
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import json
from main import VideoSubtitleExtractor, Config, SystemChecker
import logging

class VideoToSubtitleGUI:
    """图形用户界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("中文电视剧音频转文字工具 - RTX 3060 Ti专版")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 配置
        self.config = Config()
        
        # 变量
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.selected_model = tk.StringVar(value=self.config.get('preferred_model', 'faster-base'))
        self.device = tk.StringVar(value="cuda")
        self.language = tk.StringVar(value="zh")
        
        # 状态
        self.is_processing = False
        self.extractor = None
        
        self.create_widgets()
        self.setup_logging()
        
    def create_widgets(self):
        """创建界面组件"""
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="🎬 中文电视剧音频转文字工具", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="📁 文件选择", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 视频文件选择
        ttk.Label(file_frame, text="视频文件:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        video_entry = ttk.Entry(file_frame, textvariable=self.video_path, width=50)
        video_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(file_frame, text="浏览", command=self.browse_video).grid(row=0, column=2)
        
        # 输出文件选择
        ttk.Label(file_frame, text="输出字幕:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        output_entry = ttk.Entry(file_frame, textvariable=self.output_path, width=50)
        output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        ttk.Button(file_frame, text="浏览", command=self.browse_output).grid(row=1, column=2, pady=(5, 0))
        
        # 配置文件权重
        file_frame.columnconfigure(1, weight=1)
        
        # 设置区域
        settings_frame = ttk.LabelFrame(main_frame, text="⚙️ 转换设置", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 模型选择
        ttk.Label(settings_frame, text="AI模型:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        model_combo = ttk.Combobox(settings_frame, textvariable=self.selected_model, width=20)
        model_combo['values'] = ('faster-base', 'base', 'small', 'faster-large', 'medium', 'large')
        model_combo['state'] = 'readonly'
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # 设备选择
        ttk.Label(settings_frame, text="运行设备:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        device_combo = ttk.Combobox(settings_frame, textvariable=self.device, width=10)
        device_combo['values'] = ('cuda', 'cpu')
        device_combo['state'] = 'readonly'
        device_combo.grid(row=0, column=3, sticky=tk.W)
        
        # 语言选择
        ttk.Label(settings_frame, text="语言:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        language_combo = ttk.Combobox(settings_frame, textvariable=self.language, width=20)
        language_combo['values'] = ('zh', 'en', 'ja', 'ko')
        language_combo['state'] = 'readonly'
        language_combo.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # 文本后处理选项
        self.enable_postprocess = tk.BooleanVar(value=True)
        postprocess_check = ttk.Checkbutton(settings_frame, text="启用智能纠错", 
                                          variable=self.enable_postprocess)
        postprocess_check.grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=(5, 0))
        
        # 后处理说明
        postprocess_info = ttk.Label(settings_frame, text="✨ 自动修正专业名词、多音字等识别错误", 
                                   foreground="green", font=("Arial", 8))
        postprocess_info.grid(row=2, column=2, sticky=tk.W, padx=(20, 0), pady=(2, 0))
        
        # 模型说明
        model_info = ttk.Label(settings_frame, text="💡 RTX 3060 Ti推荐: faster-base (最佳平衡)", 
                              foreground="blue")
        model_info.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(10, 0))
        
        # 进度区域
        progress_frame = ttk.LabelFrame(main_frame, text="📊 处理进度", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           mode='indeterminate')
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 状态标签
        self.status_label = ttk.Label(progress_frame, text="准备就绪")
        self.status_label.grid(row=1, column=0, sticky=tk.W)
        
        # 时间标签
        self.time_label = ttk.Label(progress_frame, text="")
        self.time_label.grid(row=1, column=1, sticky=tk.E)
        
        progress_frame.columnconfigure(0, weight=1)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=(0, 10))
        
        self.start_button = ttk.Button(button_frame, text="🚀 开始转换", 
                                      command=self.start_conversion, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ 停止", 
                                     command=self.stop_conversion, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="📂 打开输出文件夹", 
                  command=self.open_output_folder).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="⚙️ 系统检查", 
                  command=self.system_check).pack(side=tk.LEFT)
        
        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="📝 日志信息", padding="10")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, width=70)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 配置主窗口权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
    def setup_logging(self):
        """设置日志输出到GUI"""
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.configure(state='disabled')
                self.text_widget.see(tk.END)
        
        # 添加GUI日志处理器
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(gui_handler)
        
    def browse_video(self):
        """浏览视频文件"""
        filetypes = [
            ("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
            ("所有文件", "*.*")
        ]
        filename = filedialog.askopenfilename(title="选择视频文件", filetypes=filetypes)
        if filename:
            self.video_path.set(filename)
            # 自动设置输出文件名
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_dir = self.config.get('output_path', './output')
            output_file = os.path.join(output_dir, f"{base_name}.srt")
            self.output_path.set(output_file)
            
    def browse_output(self):
        """浏览输出文件"""
        filename = filedialog.asksaveasfilename(
            title="保存字幕文件",
            defaultextension=".srt",
            filetypes=[("字幕文件", "*.srt"), ("所有文件", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
            
    def system_check(self):
        """系统检查"""
        def check():
            self.log_message("🔍 开始系统检查...")
            
            # 检查CUDA
            if SystemChecker.check_cuda():
                self.log_message("✅ CUDA环境正常")
            else:
                self.log_message("❌ CUDA环境异常")
                
            # 检查依赖
            if SystemChecker.check_dependencies():
                self.log_message("✅ 依赖检查通过")
            else:
                self.log_message("❌ 部分依赖缺失")
                
            self.log_message("🔍 系统检查完成")
            
        threading.Thread(target=check, daemon=True).start()
        
    def start_conversion(self):
        """开始转换"""
        if not self.video_path.get():
            messagebox.showerror("错误", "请选择视频文件")
            return
            
        if not self.output_path.get():
            messagebox.showerror("错误", "请指定输出文件")
            return
            
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()
        self.status_label.config(text="正在处理...")
        
        # 在新线程中执行转换
        thread = threading.Thread(target=self._conversion_worker, daemon=True)
        thread.start()
        
    def _conversion_worker(self):
        """转换工作线程"""
        try:
            import time
            start_time = time.time()
            
            # 创建提取器
            self.extractor = VideoSubtitleExtractor(
                model_id=self.selected_model.get(),
                device=self.device.get(),
                config=self.config
            )
            
            # 提取音频
            self.log_message(f"🎬 开始处理视频: {self.video_path.get()}")
            audio_path = self.extractor.extract_audio(self.video_path.get())
            
            if not audio_path:
                self.log_message("❌ 音频提取失败")
                return
                
            # 转录音频
            self.log_message(f"🤖 使用模型: {self.selected_model.get()}")
            result = self.extractor.transcribe_audio(
                audio_path,
                language=self.language.get(),
                temperature=0.0
            )
            
            if not result["segments"]:
                self.log_message("⚠️ 未识别到任何语音内容")
                return
                
            # 创建字幕文件
            srt_path = self.extractor.create_srt_file(
                result["segments"], 
                self.output_path.get(),
                enable_postprocess=self.enable_postprocess.get()
            )
            
            if srt_path:
                end_time = time.time()
                duration = end_time - start_time
                self.log_message(f"🎉 转换完成！耗时: {duration:.1f}秒")
                self.log_message(f"📝 共识别到 {len(result['segments'])} 个字幕片段")
                self.log_message(f"💾 文件保存至: {srt_path}")
                
                # 询问是否打开文件
                self.root.after(0, lambda: messagebox.showinfo("完成", 
                    f"转换完成！\n文件保存至: {srt_path}\n耗时: {duration:.1f}秒"))
            else:
                self.log_message("❌ 字幕文件创建失败")
                
        except Exception as e:
            self.log_message(f"❌ 处理失败: {str(e)}")
            
        finally:
            # 清理
            if self.extractor:
                self.extractor.cleanup()
                
            # 重置UI状态
            self.root.after(0, self._reset_ui)
            
    def _reset_ui(self):
        """重置UI状态"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        self.status_label.config(text="准备就绪")
        
    def stop_conversion(self):
        """停止转换"""
        self.is_processing = False
        self.log_message("⏹️ 用户取消转换")
        self._reset_ui()
        
    def open_output_folder(self):
        """打开输出文件夹"""
        output_dir = self.config.get('output_path', './output')
        if os.path.exists(output_dir):
            os.startfile(output_dir)
        else:
            messagebox.showwarning("警告", f"输出文件夹不存在: {output_dir}")
            
    def log_message(self, message):
        """添加日志消息"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)

def main():
    root = tk.Tk()
    app = VideoToSubtitleGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
