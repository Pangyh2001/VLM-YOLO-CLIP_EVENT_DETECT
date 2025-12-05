import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import csv


class DetailedLogger:
    """详细记录每个模块的处理结果"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_config = config['output']
        
        # 检查是否启用详细记录
        self.enabled = self.output_config.get('detailed_logging', False)
        
        if not self.enabled:
            print("ℹ️  Detailed logging is disabled")
            return
        
        # 创建时间戳文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self.output_config.get('output_base_dir', './output')
        self.session_dir = os.path.join(base_dir, timestamp)
        
        # 创建子目录
        self.yolo_dir = os.path.join(self.session_dir, '1_yolo_detections')
        self.clip_dir = os.path.join(self.session_dir, '2_clip_scores')
        self.vlm_dir = os.path.join(self.session_dir, '3_vlm_verifications')
        self.summary_dir = os.path.join(self.session_dir, 'summary')
        
        for dir_path in [self.yolo_dir, self.clip_dir, self.vlm_dir, self.summary_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 采样率
        self.yolo_sample_rate = self.output_config.get('yolo_sample_rate', 30)
        self.clip_sample_rate = self.output_config.get('clip_sample_rate', 10)
        
        # 记录开关
        self.save_yolo = self.output_config.get('save_yolo_detections', True)
        self.save_clip = self.output_config.get('save_clip_scores', True)
        self.save_vlm = self.output_config.get('save_vlm_results', True)
        
        # 数据记录
        self.clip_scores_data = []
        self.yolo_frame_count = 0
        self.clip_frame_count = 0
        
        # 创建CSV文件
        if self.save_clip:
            self.clip_csv_path = os.path.join(self.clip_dir, 'clip_scores.csv')
            self.clip_csv_file = open(self.clip_csv_path, 'w', newline='', encoding='utf-8')
            self.clip_csv_writer = None  # 将在第一次写入时初始化
        
        print(f"📁 Detailed logging enabled: {self.session_dir}")
    
    def log_yolo_detection(self, frame: np.ndarray, detections: List[Dict], 
                          frame_time: float, frame_idx: int):
        """记录YOLO检测结果"""
        if not self.enabled or not self.save_yolo:
            return
        
        self.yolo_frame_count += 1
        
        # 采样保存
        if self.yolo_frame_count % self.yolo_sample_rate != 0:
            return
        
        # 绘制检测框
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class']} {det['conf']:.2f}"
            if det['id'] != -1:
                label += f" ID:{det['id']}"
            
            # 绘制框
            color = self._get_class_color(det['class'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, y1-text_h-10), (x1+text_w, y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 添加信息文本
        info_text = f"Frame: {frame_idx} | Time: {frame_time:.2f}s | Detections: {len(detections)}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 保存图像
        filename = f"frame_{frame_idx:06d}_t{frame_time:.2f}s.jpg"
        filepath = os.path.join(self.yolo_dir, filename)
        cv2.imwrite(filepath, vis_frame)
        
        # 保存检测数据
        json_filename = f"frame_{frame_idx:06d}_detections.json"
        json_filepath = os.path.join(self.yolo_dir, json_filename)
        
        data = {
            'frame_idx': frame_idx,
            'frame_time': frame_time,
            'detections': detections
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def log_clip_scores(self, frame: np.ndarray, event_scores: Dict[str, float],
                       frame_time: float, frame_idx: int, detections: List[Dict]):
        """记录CLIP相似度得分"""
        if not self.enabled or not self.save_clip:
            return
        
        self.clip_frame_count += 1
        
        # 记录到CSV（所有帧）
        row_data = {
            'frame_idx': frame_idx,
            'frame_time': frame_time,
            'num_detections': len(detections)
        }
        row_data.update(event_scores)
        
        # 初始化CSV writer（第一次写入时）
        if self.clip_csv_writer is None:
            fieldnames = ['frame_idx', 'frame_time', 'num_detections'] + list(event_scores.keys())
            self.clip_csv_writer = csv.DictWriter(self.clip_csv_file, fieldnames=fieldnames)
            self.clip_csv_writer.writeheader()
        
        self.clip_csv_writer.writerow(row_data)
        self.clip_scores_data.append(row_data)
        
        # 采样保存可视化图像
        if self.clip_frame_count % self.clip_sample_rate != 0:
            return
        
        # 创建可视化
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # 创建得分面板
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # 绘制标题
        cv2.putText(panel, f"Frame {frame_idx} | Time: {frame_time:.2f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制得分条形图
        if event_scores:
            sorted_events = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)
            max_score = max(event_scores.values()) if event_scores else 1.0
            min_score = min(event_scores.values()) if event_scores else 0.0
            
            bar_width = w - 40
            bar_height = 20
            y_start = 60
            
            for i, (event_name, score) in enumerate(sorted_events):
                y = y_start + i * (bar_height + 10)
                
                # 绘制得分条
                if max_score > min_score:
                    normalized_score = (score - min_score) / (max_score - min_score)
                else:
                    normalized_score = 0.5
                
                bar_len = int(bar_width * normalized_score)
                color = (0, 255, 0) if i == 0 else (100, 100, 255)  # 最高分绿色
                
                cv2.rectangle(panel, (20, y), (20 + bar_len, y + bar_height), color, -1)
                cv2.rectangle(panel, (20, y), (20 + bar_width, y + bar_height), (150, 150, 150), 1)
                
                # 绘制文本
                text = f"{event_name}: {score:.3f}"
                cv2.putText(panel, text, (25, y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 拼接图像和面板
        combined = np.vstack([vis_frame, panel])
        
        # 保存图像
        filename = f"frame_{frame_idx:06d}_scores.jpg"
        filepath = os.path.join(self.clip_dir, filename)
        cv2.imwrite(filepath, combined)
    
    def log_vlm_verification(self, event_name: str, frames: List[np.ndarray],
                           is_confirmed: bool, reason: str, start_time: float):
        """记录VLM验证结果"""
        if not self.enabled or not self.save_vlm:
            return
        
        # 创建事件目录
        timestamp = datetime.now().strftime("%H%M%S_%f")[:12]
        event_dir = os.path.join(self.vlm_dir, f"{event_name}_{timestamp}")
        os.makedirs(event_dir, exist_ok=True)
        
        # 保存输入帧
        for i, frame in enumerate(frames):
            filename = f"input_frame_{i+1}.jpg"
            filepath = os.path.join(event_dir, filename)
            cv2.imwrite(filepath, frame)
        
        # 创建拼接图（显示所有输入帧）
        if len(frames) > 0:
            # 调整帧大小
            target_height = 240
            resized_frames = []
            for frame in frames:
                h, w = frame.shape[:2]
                target_width = int(w * target_height / h)
                resized = cv2.resize(frame, (target_width, target_height))
                resized_frames.append(resized)
            
            # 水平拼接
            if len(resized_frames) <= 4:
                combined = np.hstack(resized_frames)
            else:
                # 分两行
                row1 = np.hstack(resized_frames[:4])
                row2 = np.hstack(resized_frames[4:])
                # 补齐宽度
                if row2.shape[1] < row1.shape[1]:
                    pad_width = row1.shape[1] - row2.shape[1]
                    padding = np.zeros((row2.shape[0], pad_width, 3), dtype=np.uint8)
                    row2 = np.hstack([row2, padding])
                combined = np.vstack([row1, row2])
            
            # 添加结果标签
            result_color = (0, 255, 0) if is_confirmed else (0, 0, 255)
            result_text = "CONFIRMED" if is_confirmed else "REJECTED"
            
            label_height = 60
            label_panel = np.zeros((label_height, combined.shape[1], 3), dtype=np.uint8)
            label_panel[:] = result_color
            
            cv2.putText(label_panel, f"VLM Result: {result_text}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            combined_with_label = np.vstack([label_panel, combined])
            
            # 保存拼接图
            combined_filename = f"vlm_result_{'confirmed' if is_confirmed else 'rejected'}.jpg"
            combined_filepath = os.path.join(event_dir, combined_filename)
            cv2.imwrite(combined_filepath, combined_with_label)
        
        # 保存VLM结果JSON
        result_data = {
            'event_name': event_name,
            'start_time': start_time,
            'timestamp': datetime.now().isoformat(),
            'is_confirmed': is_confirmed,
            'reason': reason,
            'num_frames': len(frames)
        }
        
        json_filepath = os.path.join(event_dir, 'vlm_result.json')
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    def generate_summary(self):
        """生成汇总报告"""
        if not self.enabled:
            return
        
        print("📊 Generating summary report...")
        
        # 生成CLIP得分统计
        if self.save_clip and self.clip_scores_data:
            self._generate_clip_summary()
        
        # 生成总览文档
        self._generate_overview()
        
        print(f"✅ Summary saved to {self.summary_dir}")
    
    def _generate_clip_summary(self):
        """生成CLIP得分统计"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 提取数据
        frame_times = [d['frame_time'] for d in self.clip_scores_data]
        
        # 获取所有事件名称
        event_names = [k for k in self.clip_scores_data[0].keys() 
                      if k not in ['frame_idx', 'frame_time', 'num_detections']]
        
        # 绘制得分曲线
        plt.figure(figsize=(14, 8))
        
        for event_name in event_names:
            scores = [d.get(event_name, 0) for d in self.clip_scores_data]
            plt.plot(frame_times, scores, label=event_name, linewidth=2)
        
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('CLIP Similarity Score', fontsize=12)
        plt.title('CLIP Similarity Scores Over Time', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.summary_dir, 'clip_scores_timeline.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
    def log_event_crop(self, crop_img: np.ndarray, event_name: str, score: float, frame_idx: int):
        """记录送给 CLIP 的裁剪图"""
        if not self.enabled or not self.save_clip:
            return

        # 创建专门存放裁剪图的文件夹
        crop_dir = os.path.join(self.session_dir, '4_clip_crops', event_name)
        os.makedirs(crop_dir, exist_ok=True)
        
        # 文件名带上分数，方便分析
        filename = f"frame_{frame_idx:06d}_score_{score:.3f}.jpg"
        filepath = os.path.join(crop_dir, filename)
        
        # 这里的 crop_img 是 BGR 格式的 (因为是在转 RGB 之前传进来的)，可以直接保存
        cv2.imwrite(filepath, crop_img)
    
    def _generate_overview(self):
        """生成总览文档"""
        overview_path = os.path.join(self.summary_dir, 'README.md')
        
        with open(overview_path, 'w', encoding='utf-8') as f:
            f.write(f"# Event Detection Results\n\n")
            f.write(f"**Session:** {os.path.basename(self.session_dir)}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Directory Structure\n\n")
            f.write(f"```\n")
            f.write(f"{os.path.basename(self.session_dir)}/\n")
            f.write(f"├── 1_yolo_detections/     # YOLO检测结果（图像+JSON）\n")
            f.write(f"├── 2_clip_scores/         # CLIP相似度得分（图像+CSV）\n")
            f.write(f"├── 3_vlm_verifications/   # VLM验证结果（按事件分组）\n")
            f.write(f"└── summary/               # 汇总报告和统计图表\n")
            f.write(f"```\n\n")
            
            f.write(f"## Statistics\n\n")
            f.write(f"- **YOLO Detections Saved:** {self.yolo_frame_count // self.yolo_sample_rate}\n")
            f.write(f"- **CLIP Scores Recorded:** {len(self.clip_scores_data)}\n")
            
            # VLM验证统计
            if os.path.exists(self.vlm_dir):
                vlm_subdirs = [d for d in os.listdir(self.vlm_dir) 
                              if os.path.isdir(os.path.join(self.vlm_dir, d))]
                confirmed = sum(1 for d in vlm_subdirs if 'confirmed' in d.lower())
                rejected = len(vlm_subdirs) - confirmed
                
                f.write(f"- **VLM Verifications:** {len(vlm_subdirs)} total\n")
                f.write(f"  - Confirmed: {confirmed}\n")
                f.write(f"  - Rejected: {rejected}\n")
            
            f.write(f"\n## How to Use\n\n")
            f.write(f"1. **YOLO Detections**: Check `1_yolo_detections/` for object detection results\n")
            f.write(f"2. **CLIP Scores**: View `2_clip_scores/clip_scores.csv` for detailed scores\n")
            f.write(f"3. **VLM Results**: Browse `3_vlm_verifications/` to see event verification results\n")
            f.write(f"4. **Summary**: Check `summary/clip_scores_timeline.png` for score trends\n")
    
    def _get_class_color(self, class_name: str) -> tuple:
        """获取类别对应的颜色"""
        colors = {
            'person': (255, 0, 0),
            'cat': (0, 255, 255),
            'dog': (255, 255, 0),
            'door': (0, 255, 0),
            'dining table': (255, 0, 255),
            'food': (0, 165, 255),
        }
        return colors.get(class_name, (200, 200, 200))
    
    def close(self):
        """关闭日志记录器"""
        if not self.enabled:
            return
        
        # 关闭CSV文件
        if hasattr(self, 'clip_csv_file'):
            self.clip_csv_file.close()
        
        # 生成汇总
        self.generate_summary()