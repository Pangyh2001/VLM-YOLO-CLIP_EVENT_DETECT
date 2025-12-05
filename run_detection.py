#!/usr/bin/env python3
"""
视频流事件检测系统 - 主程序
支持本地视频、RTSP流、AsyncAVLoader
"""

import argparse
import yaml
import sys
import cv2
from pathlib import Path

from event_detector import EventDetector
from video_source import VideoSource


def load_config(config_path: str = "config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Video Event Detection System")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--source', type=str, required=True,
                       help='Video source: local file path, RTSP URL, or "async"')
    parser.add_argument('--display', action='store_true',
                       help='Display video with detections in real-time')
    
    # AsyncAVLoader 特定参数
    parser.add_argument('--async-mode', type=str, default='realtime',
                       choices=['realtime', 'fast'],
                       help='AsyncAVLoader mode (only if --source async)')
    parser.add_argument('--async-speed', type=float, default=1.0,
                       help='AsyncAVLoader speed multiplier')
    parser.add_argument('--async-videos', type=str, nargs='+',
                       help='Video paths for AsyncAVLoader')
    
    args = parser.parse_args()
    
    # 加载配置
    print("📋 Loading configuration...")
    config = load_config(args.config)
    
    # 创建检测器
    detector = EventDetector(config)
    
    # 创建视频源
    print(f"📹 Initializing video source: {args.source}")
    
    if args.source == 'async':
        # 使用 AsyncAVLoader
        if not args.async_videos:
            print("❌ Error: --async-videos required when using async source")
            sys.exit(1)
        
        from video_stream import AsyncAVLoader
        
        loader = AsyncAVLoader(
            video_paths=args.async_videos,
            batch_size=1,
            mode=args.async_mode,
            speed=args.async_speed,
            window_size=8,
            stride=4
        )
        video_source = VideoSource.from_async_loader(loader)
        
    elif args.source.startswith('rtsp://'):
        # RTSP 流
        video_source = VideoSource.from_rtsp(args.source)
        
    else:
        # 本地视频文件
        if not Path(args.source).exists():
            print(f"❌ Error: Video file not found: {args.source}")
            sys.exit(1)
        video_source = VideoSource.from_file(args.source)
    
    # 主处理循环
    print("\n🚀 Starting event detection...\n")
    
    try:
        for frame_data in video_source:
            frame = frame_data['frame']
            frame_time = frame_data['frame_time']
            
            # 处理帧
            result = detector.process_frame(frame, frame_time)
            
            # 如果启用了显示
            if args.display and not result.get('skipped', False):
                display_frame = frame.copy()
                
                # 绘制检测框
                for det in result.get('detections', []):
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    label = f"{det['class']} {det['conf']:.2f}"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 显示事件得分
                y_offset = 30
                for event_name, score in result.get('event_scores', {}).items():
                    text = f"{event_name}: {score:.3f}"
                    cv2.putText(display_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
                
                # 显示时间和FPS
                cv2.putText(display_frame, f"Time: {frame_time:.2f}s", (10, display_frame.shape[0]-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"FPS: {result.get('current_fps', 0)}", (10, display_frame.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Event Detection', display_frame)
                
                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏸️  User interrupted")
                    break
    
    except KeyboardInterrupt:
        print("\n⏸️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        print("\n🧹 Cleaning up...")
        
        if args.display:
            cv2.destroyAllWindows()
        
        video_source.close()
        detector.stop()
        
        print("\n✅ Done!")


if __name__ == '__main__':
    main()