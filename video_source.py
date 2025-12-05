import cv2
import numpy as np
from typing import Iterator, Dict, Any, Optional
import time


class VideoSource:
    """统一的视频源接口"""
    
    @staticmethod
    def from_file(video_path: str) -> 'LocalVideoSource':
        """从本地视频文件创建源"""
        return LocalVideoSource(video_path)
    
    @staticmethod
    def from_rtsp(rtsp_url: str) -> 'RTSPSource':
        """从 RTSP 流创建源"""
        return RTSPSource(rtsp_url)
    
    @staticmethod
    def from_async_loader(loader) -> 'AsyncLoaderSource':
        """从 AsyncAVLoader 创建源"""
        return AsyncLoaderSource(loader)


class LocalVideoSource:
    """本地视频文件源"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.fps = None
        self.frame_count = 0
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """迭代器接口"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {self.video_path}")
        print(f"   FPS: {self.fps}, Total frames: {total_frames}")
        
        self.frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                frame_time = self.frame_count / self.fps
                
                yield {
                    'frame': frame,
                    'frame_time': frame_time,
                    'frame_idx': self.frame_count,
                    'source': 'local_file'
                }
                
                self.frame_count += 1
                
        finally:
            if self.cap is not None:
                self.cap.release()
    
    def close(self):
        """关闭视频源"""
        if self.cap is not None:
            self.cap.release()


class RTSPSource:
    """RTSP 视频流源"""
    
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_count = 0
        self.start_time = None
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """迭代器接口"""
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {self.rtsp_url}")
        
        # RTSP 流可能没有准确的 FPS 信息
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25  # 默认 25 FPS
        
        print(f"📡 RTSP Stream: {self.rtsp_url}")
        print(f"   Estimated FPS: {fps}")
        
        self.frame_count = 0
        self.start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️ RTSP stream ended or error occurred")
                    break
                
                # 使用实际时间作为帧时间
                frame_time = time.time() - self.start_time
                
                yield {
                    'frame': frame,
                    'frame_time': frame_time,
                    'frame_idx': self.frame_count,
                    'source': 'rtsp'
                }
                
                self.frame_count += 1
                
        finally:
            if self.cap is not None:
                self.cap.release()
    
    def close(self):
        """关闭视频源"""
        if self.cap is not None:
            self.cap.release()


class AsyncLoaderSource:
    """基于 AsyncAVLoader 的视频源"""
    
    def __init__(self, loader):
        """
        Args:
            loader: AsyncAVLoader 实例
        """
        self.loader = loader
        self.frame_count = 0
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """迭代器接口"""
        print(f"🎬 AsyncAVLoader source started")
        
        self.frame_count = 0
        
        try:
            while True:
                batch = self.loader.queue.get()
                
                if batch is None:
                    print("📭 AsyncAVLoader ended")
                    break
                
                for item in batch:
                    if item is None:
                        continue
                    
                    # 检查是否是视频结束信号
                    if item.get('__type__') == 'video_end':
                        print(f"📹 Video ended: {item.get('source')}")
                        continue
                    
                    # 处理窗口数据
                    if item.get('type') == 'window':
                        # 取窗口的最后一帧作为当前帧
                        frames = item['frames']
                        frame = frames[-1]  # 使用窗口的最后一帧
                        frame_time = item['end_time']
                        
                        yield {
                            'frame': frame,
                            'frame_time': frame_time,
                            'frame_idx': self.frame_count,
                            'source': 'async_loader',
                            'window_frames': frames,  # 保留整个窗口供需要时使用
                            'window_start': item['start_time'],
                            'window_end': item['end_time']
                        }
                        
                        self.frame_count += 1
                    
        finally:
            self.loader.stop()
    
    def close(self):
        """关闭视频源"""
        self.loader.stop()