import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import cv2

class EventProcessor:
    """处理事件检测的核心逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.events = config['events']
        
        # 切图参数
        self.interaction_expand = config['cropping']['interaction_expand']
        self.single_expand = config['cropping']['single_expand']
        self.scene_expand = config['cropping']['scene_expand']
        
    def determine_event_type(self, event_name: str) -> str:
        """根据事件名称获取事件类型"""
        for event in self.events:
            if event['name'] == event_name:
                return event['type']
        return 'scene'  # 默认
    
    def crop_image_by_event(self, frame: np.ndarray, detections: List[Dict],
                           event: Dict) -> Optional[np.ndarray]:
        """
        根据事件类型和检测结果裁剪图像
        """
        event_type = event['type']
        h, w = frame.shape[:2]
        
        if event_type == 'location':
            # 位置关系：不需要裁剪，直接返回原图（后续会计算IoU）
            return frame
            
        elif event_type == 'interaction':
            # 互动类：找到所有相关实体的框，计算最小外接矩形
            relevant_boxes = []
            for det in detections:
                if det['class'] in event['entities']:
                    relevant_boxes.append(det['bbox'])
            
            if len(relevant_boxes) < 2:
                # 至少需要两个实体才能互动
                return None
                
            # 计算最小外接矩形
            all_x1 = [box[0] for box in relevant_boxes]
            all_y1 = [box[1] for box in relevant_boxes]
            all_x2 = [box[2] for box in relevant_boxes]
            all_y2 = [box[3] for box in relevant_boxes]
            
            x1 = min(all_x1)
            y1 = min(all_y1)
            x2 = max(all_x2)
            y2 = max(all_y2)
            
            # 扩展
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            new_w = width * self.interaction_expand
            new_h = height * self.interaction_expand
            
            x1 = max(0, int(cx - new_w / 2))
            y1 = max(0, int(cy - new_h / 2))
            x2 = min(w, int(cx + new_w / 2))
            y2 = min(h, int(cy + new_h / 2))
            
            return frame[y1:y2, x1:x2]
            
        elif event_type == 'single':
            # 单人动作：只裁剪人
            person_boxes = [det['bbox'] for det in detections if det['class'] == 'person']
            
            if not person_boxes:
                return None
            
            # 选择第一个人（也可以选择最大的）
            box = person_boxes[0]
            x1, y1, x2, y2 = box
            
            # 扩展
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            new_w = width * self.single_expand
            new_h = height * self.single_expand
            
            x1 = max(0, int(cx - new_w / 2))
            y1 = max(0, int(cy - new_h / 2))
            x2 = min(w, int(cx + new_w / 2))
            y2 = min(h, int(cy + new_h / 2))
            
            return frame[y1:y2, x1:x2]
            
        elif event_type == 'scene':
            # 场景类：找主要物体（如餐桌）为中心扩展，或直接用全图
            main_entity = event['entities'][1] if len(event['entities']) > 1 else event['entities'][0]
            main_boxes = [det['bbox'] for det in detections if det['class'] == main_entity]
            
            if not main_boxes:
                # 没有主要物体，返回全图
                return frame
            
            # 以主要物体为中心扩展
            box = main_boxes[0]
            x1, y1, x2, y2 = box
            
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            new_w = width * self.scene_expand
            new_h = height * self.scene_expand
            
            x1 = max(0, int(cx - new_w / 2))
            y1 = max(0, int(cy - new_h / 2))
            x2 = min(w, int(cx + new_w / 2))
            y2 = min(h, int(cy + new_h / 2))
            
            return frame[y1:y2, x1:x2]
        
        return frame
    
    def check_location_event(self, detections: List[Dict], event: Dict) -> bool:
        """
        检查位置关系事件（如进出门）
        通过计算两个实体的 IoU 来判断
        """
        if event['type'] != 'location':
            return False
        
        entities = event['entities']
        if len(entities) != 2:
            return False
        
        # 找到两类实体的框
        entity1_boxes = [det['bbox'] for det in detections if det['class'] == entities[0]]
        entity2_boxes = [det['bbox'] for det in detections if det['class'] == entities[1]]
        
        if not entity1_boxes or not entity2_boxes:
            return False
        
        # 计算任意两个框的 IoU
        for box1 in entity1_boxes:
            for box2 in entity2_boxes:
                iou = self._compute_iou(box1, box2)
                if iou > 0.1:  # 有重叠就认为在进行位置相关的事件
                    return True
        
        return False
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个框的 IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class EventTracker:
    """跟踪每个事件的状态"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_start_frames = config['detection']['event_start_frames']
        self.event_end_frames = config['detection']['event_end_frames']
        self.clip_consecutive_frames = config['detection']['clip_consecutive_frames']
        
        # 每个事件的状态
        self.event_states = {}  # {event_name: EventState}
        
    def init_event(self, event_name: str):
        """初始化事件状态"""
        if event_name not in self.event_states:
            self.event_states[event_name] = EventState(
                event_name, 
                self.event_start_frames,
                self.event_end_frames,
                self.clip_consecutive_frames
            )
    
    def update_event(self, event_name: str, is_highest: bool, 
                    similarity: float, frame_time: float, 
                    cropped_frame: Optional[np.ndarray]) -> Tuple[Optional[str], Optional[Dict]]:
        """
        更新事件状态
        返回: (触发类型, 数据)
          - ('vlm_check', {'frames': [...], 'event_name': ...}): 需要VLM验证
          - ('event_start', {'event_name': ..., 'start_time': ...}): 事件开始
          - ('event_end', {'event_name': ..., 'end_time': ...}): 事件结束
          - (None, None): 无特殊情况
        """
        state = self.event_states[event_name]
        return state.update(is_highest, similarity, frame_time, cropped_frame)
    
    def vlm_confirmed(self, event_name: str, confirmed: bool):
        """VLM 确认结果"""
        if event_name in self.event_states:
            self.event_states[event_name].vlm_confirmed(confirmed)
    
    def get_active_events(self) -> List[str]:
        """获取当前活跃的事件"""
        return [name for name, state in self.event_states.items() if state.is_active]


class EventState:
    """单个事件的状态"""
    
    def __init__(self, name: str, start_frames: int, end_frames: int, clip_frames: int):
        self.name = name
        self.start_frames = start_frames
        self.end_frames = end_frames
        self.clip_frames = clip_frames
        
        self.is_active = False
        self.start_time = None
        self.pending_vlm = False
        
        # 用于判断事件开始的缓冲
        self.high_score_buffer = deque(maxlen=clip_frames)  # 存储是否是最高分
        self.start_candidate_buffer = deque(maxlen=start_frames)  # 存储时间和帧
        
        # 用于判断事件结束的缓冲
        self.low_score_buffer = deque(maxlen=end_frames)
        
    def update(self, is_highest: bool, similarity: float, frame_time: float,
              cropped_frame: Optional[np.ndarray]) -> Tuple[Optional[str], Optional[Dict]]:
        """更新状态"""
        
        if not self.is_active:
            # 尝试检测事件开始
            self.high_score_buffer.append(is_highest)
            self.start_candidate_buffer.append({
                'time': frame_time,
                'frame': cropped_frame,
                'similarity': similarity
            })
            
            # 检查是否连续N帧都是最高分
            if len(self.high_score_buffer) == self.clip_frames and all(self.high_score_buffer):
                # 触发 VLM 验证
                if not self.pending_vlm:
                    self.pending_vlm = True
                    frames = [item['frame'] for item in self.start_candidate_buffer 
                             if item['frame'] is not None]
                    return ('vlm_check', {
                        'frames': frames,
                        'event_name': self.name,
                        'start_time': self.start_candidate_buffer[0]['time']
                    })
        else:
            # 事件已激活，检查是否结束
            self.low_score_buffer.append(not is_highest)
            
            # 连续N帧都不是最高分，事件结束
            if len(self.low_score_buffer) == self.end_frames and all(self.low_score_buffer):
                end_time = frame_time
                self.is_active = False
                self.start_time = None
                self.low_score_buffer.clear()
                
                return ('event_end', {
                    'event_name': self.name,
                    'end_time': end_time
                })
        
        return (None, None)
    
    def vlm_confirmed(self, confirmed: bool):
        """VLM 确认后调用"""
        self.pending_vlm = False
        
        if confirmed and not self.is_active:
            # 事件开始
            self.is_active = True
            self.start_time = self.start_candidate_buffer[0]['time'] if self.start_candidate_buffer else None
            self.low_score_buffer.clear()