import subprocess
import numpy as np
import cv2
import os

class RTSPVideoStream:
    def __init__(self, rtsp_url, ffmpeg_path="/usr/local/ffmpeg-4.4.5/bin/ffmpeg"):
        self.rtsp_url = rtsp_url
        self.ffmpeg_path = ffmpeg_path
        self.width, self.height = self.get_resolution()
        if not self.width or not self.height:
            raise ValueError("❌ 无法获取 RTSP 视频分辨率")
        self.frame_size = self.width * self.height * 3
        self.process = None

    def get_resolution(self):
        """使用 ffprobe 获取 RTSP 视频分辨率"""
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0', self.rtsp_url
        ]
        try:
            output = subprocess.check_output(cmd).decode().strip()
            width, height = map(int, output.split(','))
            return width, height
        except Exception as e:
            print('❌ 获取分辨率失败:', e)
            return None, None

    def start(self):
        """启动 FFmpeg 拉流"""
        ffmpeg_cmd = [
            self.ffmpeg_path, # 指定ffmpeg路径
             "-loglevel", "quiet", # 静音模式 （不输出日志）
            "-rtsp_transport", "tcp", # 使用TCP模式，更稳定
            "-i", self.rtsp_url, # 输入RTSP视频流
            "-f", "image2pipe", # 输出格式为图像数据流
            "-pix_fmt", "bgr24", # 每个像素用RGB表示 （OpenCV 默认格式）
            "-vcodec", "rawvideo", # 原始未压缩图像数据
            "-" # 输出到标准输出 stdout
        ]
        self.process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
        print("🚀 FFmpeg 拉流已启动")

    def read(self):
        """读取一帧图像"""
        if not self.process:
            raise RuntimeError("必须先调用 start()")
        raw_frame = self.process.stdout.read(self.frame_size)
        if len(raw_frame) != self.frame_size:
            return None
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame

    def stop(self):
        """关闭 FFmpeg 进程"""
        if self.process:
            self.process.kill()
            self.process = None
            print("🛑 拉流已关闭")