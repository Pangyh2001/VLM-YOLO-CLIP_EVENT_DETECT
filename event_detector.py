import numpy as np
import time
import json
import os
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import cv2

from models import ModelManager
from event_processor import EventProcessor, EventTracker
from vlm_worker import VLMWorkerPool
from result_logger import DetailedLogger


class EventDetector:
    """视频流事件检测主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化模型
        print("🚀 Initializing Event Detector...")
        self.model_manager = ModelManager(config)
        
        # 初始化事件处理器
        self.event_processor = EventProcessor(config)
        self.event_tracker = EventTracker(config)
        
        # 初始化所有事件
        for event in config['events']:
            self.event_tracker.init_event(event['name'])
        
        # 提取所有需要检测的实体（去重）
        all_entities = set()
        for event in config['events']:
            all_entities.update(event['entities'])
        self.all_entities = list(all_entities)
        
        # 设置 YOLO 检测类别
        self.model_manager.set_yolo_classes(self.all_entities)
        print(f"📋 Monitoring entities: {self.all_entities}")
        
        # 帧率控制
        self.base_fps = config['video']['base_fps']
        self.active_fps = config['video']['active_fps']
        self.no_detection_frames_threshold = config['video']['no_detection_frames']
        self.current_fps = self.base_fps
        self.no_detection_count = 0
        
        # 检测结果记录
        self.detection_results = []
        
        # 输出配置
        self.save_json = config['output']['save_json']
        self.json_path = config['output']['json_path']
        self.save_event_images = config['output']['save_event_images']
        self.event_images_dir = config['output']['event_images_dir']
        
        if self.save_event_images:
            os.makedirs(self.event_images_dir, exist_ok=True)
        
        # 初始化详细日志记录器
        self.logger = DetailedLogger(config)
        
        # 初始化 VLM 线程池
        self.vlm_pool = VLMWorkerPool(
            self.model_manager, 
            config, 
            self._vlm_callback,
            self.logger  # 传递logger
        )
        
        # 统计信息
        self.frame_count = 0
        self.processed_frame_count = 0
        self.last_stats_time = time.time()
        
        print("✅ Event Detector initialized")
    
    def _vlm_callback(self, event_name: str, is_confirmed: bool, reason: str):
        """VLM 验证完成后的回调"""
        self.event_tracker.vlm_confirmed(event_name, is_confirmed)
        
        if is_confirmed:
            # 记录事件开始
            state = self.event_tracker.event_states[event_name]
            result = {
                'event_name': event_name,
                'start_time': state.start_time,
                'end_time': None,
                'status': 'ongoing',
                'vlm_reason': reason
            }
            self.detection_results.append(result)
            
            print(f"🎯 Event STARTED: {event_name} at {state.start_time:.2f}s")
            print(f"   VLM reason: {reason}")
    
    def _should_process_frame(self, frame_idx: int) -> bool:
        """判断是否应该处理当前帧（基于动态帧率）"""
        if self.current_fps >= self.active_fps:
            # 高帧率模式或帧率相同：每帧都处理
            return True
        else:
            # 低帧率模式：按比例采样
            sample_rate = self.current_fps / self.active_fps
            if sample_rate <= 0:
                return True  # 防止除零错误
            skip_interval = int(1.0 / sample_rate)
            if skip_interval <= 0:
                return True
            return (frame_idx % skip_interval) == 0
    
    def process_frame(self, frame: np.ndarray, frame_time: float) -> Dict[str, Any]:
        """
        处理单帧
        返回: 处理结果字典
        """
        self.frame_count += 1
        
        # 动态帧率控制
        if not self._should_process_frame(self.frame_count):
            return {'skipped': True}
        
        self.processed_frame_count += 1
        start_time = time.time()
        
        # Step 1: YOLO 检测
        detections = self.model_manager.detect_objects(frame)
        
        # 记录YOLO检测结果
        self.logger.log_yolo_detection(frame, detections, frame_time, self.frame_count)
        
        # 更新帧率状态
        if len(detections) > 0:
            if self.current_fps != self.active_fps:
                self.current_fps = self.active_fps
                print(f"⚡ FPS increased to {self.active_fps}")
            self.no_detection_count = 0
        else:
            self.no_detection_count += 1
            if self.no_detection_count >= self.no_detection_frames_threshold:
                if self.current_fps != self.base_fps:
                    self.current_fps = self.base_fps
                    print(f"🐌 FPS decreased to {self.base_fps}")
        
        # 如果没有检测到任何实体，跳过后续处理
        if len(detections) == 0:
            return {
                'frame_time': frame_time,
                'detections': [],
                'events': {},
                'processing_time': time.time() - start_time
            }
        
        # Step 2: 对每个事件进行处理
        event_scores = {}
        
        for event in self.config['events']:
            event_name = event['name']
            
            # 检查是否有所需的实体
            detected_entities = set(det['class'] for det in detections)
            required_entities = set(event['entities'])
            
            # 宽松模式：只要检测到了任意一个相关实体，就放行
            # isdisjoint() 如果交集为空返回 True，所以这里的意思是“如果一个重合的都没有，才跳过”
            if required_entities.isdisjoint(detected_entities):
                event_scores[event_name] = 0.0
                continue
            
            # 特殊处理位置关系事件
            if event['type'] == 'location':
                if self.event_processor.check_location_event(detections, event):
                    # 对于位置事件，直接用原图计算 CLIP
                    cropped = frame
                else:
                    event_scores[event_name] = 0.0
                    continue
            else:
                # 根据事件类型裁剪图像
                cropped = self.event_processor.crop_image_by_event(frame, detections, event)
                
                if cropped is None:
                    event_scores[event_name] = 0.0
                    continue
            
            # Step 3: CLIP 计算相似度
            # 与正面描述对比
            pos_score = self.model_manager.compute_clip_similarity(
                cropped, event['positive_desc']
            )
            
            # 与负面描述对比（取最大值）
            neg_scores = [
                self.model_manager.compute_clip_similarity(cropped, neg_desc)
                for neg_desc in event['negative_descs']
            ]
            max_neg_score = max(neg_scores) if neg_scores else 0.0
            
            # 最终分数：正面分数减去负面分数
            final_score = pos_score - max_neg_score
            event_scores[event_name] = max(0.0, final_score)
            
            
            
            # --- 【新增】保存 CLIP 看到的裁剪图 ---
            # 只有当分数超过一定阈值（比如认为可能是事件）时才保存，节省磁盘
            if final_score > 0.1: 
                self.logger.log_event_crop(cropped, event_name, final_score, self.frame_count)
            # -----------------------------------
            
            
            
            # Step 4: 更新事件状态
            is_highest = final_score == max(event_scores.values())
            
            trigger_type, trigger_data = self.event_tracker.update_event(
                event_name, is_highest, final_score, frame_time, cropped
            )
            
            # 处理触发
            if trigger_type == 'vlm_check':
                # 提交 VLM 验证任务
                self.vlm_pool.submit_task(
                    event_name,
                    trigger_data['frames'],
                    event['positive_desc'],
                    trigger_data['start_time']
                )
                
            elif trigger_type == 'event_end':
                # 事件结束
                self._record_event_end(event_name, frame_time)
        
        # 记录CLIP得分
        self.logger.log_clip_scores(frame, event_scores, frame_time, self.frame_count, detections)
        
        # 统计信息
        processing_time = time.time() - start_time
        
        if time.time() - self.last_stats_time > 5.0:
            self._print_stats()
            self.last_stats_time = time.time()
        
        return {
            'frame_time': frame_time,
            'detections': detections,
            'event_scores': event_scores,
            'processing_time': processing_time,
            'current_fps': self.current_fps
        }
    
    def _record_event_end(self, event_name: str, end_time: float):
        """记录事件结束"""
        # 找到最近的未结束的该事件
        for result in reversed(self.detection_results):
            if result['event_name'] == event_name and result['end_time'] is None:
                result['end_time'] = end_time
                result['status'] = 'completed'
                result['duration'] = end_time - result['start_time']
                
                print(f"🏁 Event ENDED: {event_name} at {end_time:.2f}s "
                      f"(duration: {result['duration']:.2f}s)")
                break
    
    def _print_stats(self):
        """打印统计信息"""
        vlm_stats = self.vlm_pool.get_stats()
        active_events = self.event_tracker.get_active_events()
        
        print(f"\n📊 Statistics:")
        print(f"   Frames: {self.frame_count} | Processed: {self.processed_frame_count}")
        print(f"   Current FPS: {self.current_fps}")
        print(f"   Active Events: {active_events if active_events else 'None'}")
        print(f"   VLM Queue: {self.vlm_pool.get_queue_size()} | "
              f"Total: {vlm_stats['total_tasks']} | "
              f"Completed: {vlm_stats['completed_tasks']} | "
              f"Confirmed: {vlm_stats['confirmed_events']} | "
              f"Rejected: {vlm_stats['rejected_events']}")
        print()
    
    def save_results(self):
        """保存检测结果到 JSON"""
        if not self.save_json:
            return
        
        output = {
            'metadata': {
                'total_frames': self.frame_count,
                'processed_frames': self.processed_frame_count,
                'events_config': self.config['events']
            },
            'results': self.detection_results
        }
        
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {self.json_path}")
    
    def stop(self):
        """停止检测器"""
        print("\n🛑 Stopping Event Detector...")
        self.vlm_pool.stop()
        self.logger.close()
        self.save_results()
        print("✅ Event Detector stopped")