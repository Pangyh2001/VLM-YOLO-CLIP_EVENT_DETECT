import threading
import queue
from typing import Dict, Any, Callable, List
import numpy as np
import time

class VLMWorkerPool:
    """VLM 验证的异步线程池"""
    
    def __init__(self, model_manager, config: Dict[str, Any], 
                 callback: Callable[[str, bool, str], None],
                 logger=None):
        """
        Args:
            model_manager: ModelManager 实例
            config: 配置字典
            callback: 回调函数 callback(event_name, is_confirmed, reason)
            logger: DetailedLogger 实例（可选）
        """
        self.model_manager = model_manager
        self.config = config
        self.callback = callback
        self.logger = logger
        
        self.max_workers = config['vlm']['max_workers']
        self.task_queue = queue.Queue(maxsize=config['vlm']['queue_size'])
        
        self.stop_event = threading.Event()
        self.workers = []
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'confirmed_events': 0,
            'rejected_events': 0
        }
        self.stats_lock = threading.Lock()
        
        # 启动工作线程
        for i in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self.workers.append(t)
        
        print(f"✅ VLM Worker Pool started with {self.max_workers} workers")
    
    def submit_task(self, event_name: str, frames: List[np.ndarray], 
                   positive_desc: str, start_time: float) -> bool:
        """
        提交 VLM 验证任务
        返回: 是否成功提交（队列未满）
        """
        task = {
            'event_name': event_name,
            'frames': frames,
            'positive_desc': positive_desc,
            'start_time': start_time,
            'submit_time': time.time()
        }
        
        try:
            self.task_queue.put(task, block=False)
            with self.stats_lock:
                self.stats['total_tasks'] += 1
            return True
        except queue.Full:
            print(f"⚠️ VLM queue is full, dropping task for event: {event_name}")
            return False
    
    def _worker_loop(self, worker_id: int):
        """工作线程的主循环"""
        while not self.stop_event.is_set():
            try:
                # 获取任务（超时1秒）
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:  # 停止信号
                    break
                
                # 执行 VLM 验证
                self._process_task(worker_id, task)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Worker-{worker_id} error: {e}")
    
    def _process_task(self, worker_id: int, task: Dict[str, Any]):
        """处理单个验证任务"""
        event_name = task['event_name']
        frames = task['frames']
        positive_desc = task['positive_desc']
        
        start = time.time()
        
        try:
            # 调用 VLM 进行验证
            is_confirmed, reason = self.model_manager.vlm_verify_event(
                frames, event_name, positive_desc
            )
            
            elapsed = time.time() - start
            wait_time = time.time() - task['submit_time']
            
            print(f"🔍 Worker-{worker_id} | Event: {event_name} | "
                  f"Result: {'✅ CONFIRMED' if is_confirmed else '❌ REJECTED'} | "
                  f"VLM time: {elapsed:.2f}s | Wait time: {wait_time:.2f}s")
            
            # 更新统计
            with self.stats_lock:
                self.stats['completed_tasks'] += 1
                if is_confirmed:
                    self.stats['confirmed_events'] += 1
                else:
                    self.stats['rejected_events'] += 1
            
            # 调用回调函数
            self.callback(event_name, is_confirmed, reason)
            
        except Exception as e:
            print(f"❌ VLM verification failed for {event_name}: {e}")
            self.callback(event_name, False, f"Error: {str(e)}")
    
    def stop(self):
        """停止所有工作线程"""
        print("🛑 Stopping VLM Worker Pool...")
        self.stop_event.set()
        
        # 发送停止信号
        for _ in range(self.max_workers):
            try:
                self.task_queue.put(None, block=False)
            except queue.Full:
                pass
        
        # 等待线程结束
        for t in self.workers:
            t.join(timeout=2.0)
        
        print("✅ VLM Worker Pool stopped")
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self.stats_lock:
            return self.stats.copy()
    
    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return self.task_queue.qsize()