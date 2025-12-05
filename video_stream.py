import av
import time
import numpy as np
import threading
import queue
from typing import Iterator, Dict, Any, List, Union
from collections import deque

# 在需要测试以下内容的时候，才需要模拟实时视频流，其他时候加速模式就行了，这时候消费者决定了生产者的速度。
'''
1. 端到端延迟
比如：事件在 t=10s 发生，你的系统在 t=10.6s 才报警，这个 0.6s 的“告警延迟”就和真实时间有关，这时需要用 realtime 模式去测。

2. 系统级压测 / 吞吐量
看看在 15FPS、2560×1440 的条件下，你整套系统（解码 + 推理 + 后处理）能不能跟得上，不丢帧。

3. 和真实摄像头 / 上层平台对接
对接 RTSP/摄像头/流媒体服务器，需要保证接口形式就是“源源不断来的帧/窗口”，这时流式 loader 有用。

'''

def stream_av_from_video(
    video_path: str, 
    mode: str = "realtime", 
    speed: float = 1.0,
    window_size: int = 8,  # 窗口大小（帧数）
    stride: int = 4        # 步长（滑动多少帧）
) -> Iterator[Dict[str, Any]]:
    """
    生成器：从视频读取帧，并按滑动窗口输出
    """
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"❌ Error opening video {video_path}: {e}")
        return

    if not container.streams.video:
        return

    video_stream = container.streams.video[0]
    internal_frame_count = 0
    start_wall_time = time.time()
    video_start_pts = None
    
    # 窗口缓冲区
    frame_buffer = [] # 存储 {'frame': img, 'pts': ts}

    try:
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if video_start_pts is None:
                    video_start_pts = float(frame.pts * video_stream.time_base)
                
                v_pts = float(frame.pts * video_stream.time_base)
                frame_img = frame.to_ndarray(format="bgr24")
                
                # 实时流模拟
                if mode == "realtime":
                    rel_pts = v_pts - video_start_pts
                    target_wall = start_wall_time + rel_pts / speed
                    now = time.time()
                    if target_wall > now:
                        time.sleep(target_wall - now)

                # 加入缓冲区
                frame_buffer.append({
                    'frame': frame_img,
                    'pts': v_pts,
                    'index': internal_frame_count
                })
                internal_frame_count += 1

                # === 滑动窗口逻辑 ===
                if len(frame_buffer) >= window_size:
                    # 提取窗口数据
                    window_frames = [x['frame'] for x in frame_buffer]
                    start_ts = frame_buffer[0]['pts']
                    end_ts = frame_buffer[-1]['pts']
                    
                    yield {
                        "source": video_path,
                        "type": "window", # 标记为窗口数据
                        "frames": window_frames, # List[np.ndarray]
                        "start_time": start_ts,
                        "end_time": end_ts,
                        "video_pts": end_ts, # 兼容旧逻辑，取窗口结束时间
                        "enqueue_time": time.time()
                    }

                    # 滑动：移除 stride 个旧帧
                    # 保持 buffer 中剩余 window_size - stride 个帧，作为下一个窗口的开头
                    frame_buffer = frame_buffer[stride:]

    except Exception as e:
        print(f"⚠️ Decode error in {video_path}: {e}")
    finally:
        container.close()


class AsyncAVLoader:
    def __init__(self, video_paths: Union[str, List[str]], batch_size: int = 1, 
                 mode="realtime", speed=1.0, queue_size=100,
                 window_size: int = 8, stride: int = 4): # 新增参数
        
        if isinstance(video_paths, str):
            self.video_paths = [video_paths]
        else:
            self.video_paths = list(video_paths)
            
        self.batch_size = batch_size
        self.mode = mode
        self.speed = speed
        self.window_size = window_size
        self.stride = stride
        
        self.pending_videos = self.video_paths.copy()
        self.pending_lock = threading.Lock()
        
        self.queue = queue.Queue(maxsize=queue_size)
        self.internal_queues = [queue.Queue(maxsize=10) for _ in range(self.batch_size)]
        
        self.stop_event = threading.Event()
        self.workers = []
        
        for i in range(self.batch_size):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self.workers.append(t)
            
        self.aggregator_thread = threading.Thread(target=self._aggregator, daemon=True)
        self.aggregator_thread.start()
    
    def _get_next_video(self):
        with self.pending_lock:
            if not self.pending_videos:
                return None
            return self.pending_videos.pop(0)

    def _worker_loop(self, index):
        try:
            while not self.stop_event.is_set():
                video_path = self._get_next_video()
                if video_path is None:
                    break
                
                # 传入窗口参数
                iterator = stream_av_from_video(
                    video_path, self.mode, self.speed, 
                    self.window_size, self.stride
                )
                
                for pkt in iterator:
                    if self.stop_event.is_set():
                        return
                    self.internal_queues[index].put(pkt)
                
                # 发送视频结束信号
                self.internal_queues[index].put({
                    '__type__': 'video_end', 
                    'source': video_path
                })
                
        except Exception as e:
            print(f"Worker-{index} error: {e}")
        finally:
            self.internal_queues[index].put(None)

    def _aggregator(self):
        streams_alive = [True] * self.batch_size
        
        while not self.stop_event.is_set():
            if not any(streams_alive):
                break
            
            batch_items = []
            
            for i in range(self.batch_size):
                if not streams_alive[i]:
                    batch_items.append(None)
                    continue
                
                q = self.internal_queues[i]
                item = q.get()
                
                if item is None:
                    streams_alive[i] = False
                    batch_items.append(None)
                else:
                    batch_items.append(item)
            
            if all(item is None for item in batch_items):
                break
                
            self.queue.put(batch_items)
            
        self.queue.put(None)

    def stop(self):
        self.stop_event.set()
        if self.aggregator_thread.is_alive():
            self.aggregator_thread.join(timeout=1.0)
        for t in self.workers:
            if t.is_alive():
                t.join(timeout=0.5)