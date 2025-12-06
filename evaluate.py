#!/usr/bin/env python3
import os
import random
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp
import math
import sys

# --- 配置部分 ---
DATASET_ROOT = "/data2/pyh/video_stream_event_detection/zhongjifangan/dataset/easy_data"
CONFIG_PATH = "config.yaml"
SAMPLE_SIZE = 500
OUTPUT_DIR = "./evaluation_results"
GPU_IDS = [4, 6, 7]  # 使用的GPU编号

GT_MAPPING = {
    "Enter": "Entering Exiting",
    "Exit": "Entering Exiting",
    "petting_cat": "Petting Cat",
    "dining": "Dining",
    "eating_cake": "Eating Cake",
    "carrying_baby": "Carrying Baby",
    "crawling_baby": "Baby Crawling",
    "Hand_clap": "Clapping",
    "Hand_wave": "Waving",
    "petting_animal_(not_cat)": "Petting Animal",
    "walking_the_dog": "Walking Dog"
}

def load_and_patch_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if 'output' in config:
        config['output']['save_json'] = True 
        config['output']['save_event_images'] = False 
        config['output']['detailed_logging'] = False  
    return config

def get_test_dataset(root_dir, sample_size):
    all_videos = []
    root = Path(root_dir)
    for folder in root.iterdir():
        if folder.is_dir() and folder.name in GT_MAPPING:
            gt = GT_MAPPING[folder.name]
            videos = list(folder.glob("*.mp4")) + list(folder.glob("*.avi"))
            for v in videos:
                all_videos.append({"path": str(v), "folder": folder.name, "gt_event": gt})
    
    if len(all_videos) > sample_size:
        return random.sample(all_videos, sample_size)
    return all_videos

# --- 核心工作函数 ---
def worker_process(gpu_id, video_subset, config, output_queue):
    """
    每个GPU上运行的工作进程
    """
    try:
        # 1. 先设置显卡可见性，再导入 torch
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 延迟导入，防止主进程初始化CUDA
        from event_detector import EventDetector
        from video_source import VideoSource
        from event_processor import EventTracker # 需要导入这个来重置状态
        import torch

        print(f"🚀 GPU {gpu_id}: Initialized. Processing {len(video_subset)} videos")
        
        # 2. 【关键修改】在循环外只初始化一次检测器
        # 这样模型只加载一次，显存占用稳定在 15GB 左右
        detector = EventDetector(config)
        
        # 预设置 YOLO 类别
        all_entities = set()
        for event in config['events']:
            all_entities.update(event['entities'])
        detector.model_manager.set_yolo_classes(list(all_entities))
        
        local_metrics = {"total": 0, "correct": 0, "miss": 0, "false_alarm": 0}
        local_results = []
        
        # 定义内部处理函数
        def process_video_stream(det, v_path):
            v_source = VideoSource.from_file(v_path)
            detected = set()
            try:
                for fd in v_source:
                    det.process_frame(fd['frame'], fd['frame_time'])
                
                # 收集 ongoing 和 completed 的事件
                for res in det.detection_results:
                    if res.get('status') in ['ongoing', 'completed']:
                        detected.add(res['event_name'])
            finally:
                v_source.close()
            return detected

        # 定义重置状态函数
        def reset_detector_state(det):
            # 清空结果列表
            det.detection_results = []
            det.frame_count = 0
            det.processed_frame_count = 0
            det.current_fps = det.base_fps
            det.no_detection_count = 0
            
            # 重置事件追踪器 (创建一个新的 Tracker 实例最干净)
            det.event_tracker = EventTracker(det.config)
            for evt in det.config['events']:
                det.event_tracker.init_event(evt['name'])

        # 3. 循环处理视频
        iterator = tqdm(video_subset, desc=f"GPU {gpu_id}", position=gpu_id) if len(video_subset) > 0 else video_subset
        
        for video_info in iterator:
            local_metrics["total"] += 1
            video_path = video_info['path']
            video_name = Path(video_path).stem 
            
            # A. 重置检测器状态 (复用实例)
            reset_detector_state(detector)
            
            # B. 运行检测
            detected_set = process_video_stream(detector, video_path)
            
            # C. 等待 VLM 异步任务全部完成 (关键!)
            # 必须等待上一条视频的 VLM 判决做完，防止串台
            detector.vlm_pool.task_queue.join() 
            
            # 补充检查：如果有刚才 VLM 刚确认的事件，加进来
            for res in detector.detection_results:
                if res.get('status') in ['ongoing', 'completed']:
                    detected_set.add(res['event_name'])

            # D. 保存结果
            json_filename = f"{video_name}_result.json"
            save_path = os.path.join(OUTPUT_DIR, json_filename)
            detector.save_results(output_path=save_path)
            
            # E. 【关键】不要调用 detector.stop()，否则线程池会死掉
            # 我们只在所有视频跑完后 stop 一次
            
            # F. 统计
            gt_event = video_info['gt_event']
            is_correct = False
            status = ""
            
            if len(detected_set) == 0:
                local_metrics["miss"] += 1
                status = "⭕ Miss"
            elif gt_event in detected_set:
                local_metrics["correct"] += 1
                is_correct = True
                status = "✅"
            else:
                local_metrics["false_alarm"] += 1
                status = "❌ False"
                
            if not is_correct:
                local_results.append(f"{status} | GT: {gt_event:<15} | Det: {list(detected_set)} | {video_name}")
        
        # 4. 所有视频跑完，彻底释放资源
        detector.stop(should_save=False)
        
        # 发回结果
        output_queue.put((gpu_id, local_metrics, local_results))
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} Error: {e}")
        import traceback
        traceback.print_exc()
        # 发送空结果防止主进程死锁
        output_queue.put((gpu_id, {"total":0,"correct":0,"miss":0,"false_alarm":0}, []))

def main():
    # 设置多进程启动方式
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 清理环境变量
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    config = load_and_patch_config(CONFIG_PATH)
    test_videos = get_test_dataset(DATASET_ROOT, SAMPLE_SIZE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 将视频分配到不同GPU
    num_gpus = len(GPU_IDS)
    if num_gpus == 0: return

    # 均匀切分
    chunk_size = math.ceil(len(test_videos) / num_gpus)
    video_subsets = [test_videos[i:i + chunk_size] for i in range(0, len(test_videos), chunk_size)]
    
    print(f"📊 Distribution:")
    for i in range(len(video_subsets)):
        print(f"  GPU {GPU_IDS[i]}: {len(video_subsets[i])} videos")
    
    # 创建进程和队列
    output_queue = mp.Queue()
    processes = []
    
    for i in range(len(video_subsets)):
        gpu_id = GPU_IDS[i]
        p = mp.Process(target=worker_process, 
                       args=(gpu_id, video_subsets[i], config, output_queue))
        p.start()
        processes.append(p)
    
    # 收集结果
    total_metrics = {"total": 0, "correct": 0, "miss": 0, "false_alarm": 0}
    all_results_log = []
    finished_count = 0
    
    while finished_count < len(processes):
        gpu_id, metrics, results = output_queue.get()
        print(f"\n📊 GPU {gpu_id} finished.")
        
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        all_results_log.extend(results)
        finished_count += 1
    
    for p in processes:
        p.join()
    
    # 打印总体报告
    print("\n" + "="*60)
    print("📊 Overall Evaluation Report")
    print("="*60)
    total = total_metrics['total']
    if total > 0:
        print(f"Total: {total}")
        print(f"Correct: {total_metrics['correct']} ({total_metrics['correct']/total*100:.2f}%)")
        print(f"Miss: {total_metrics['miss']} ({total_metrics['miss']/total*100:.2f}%)")
        print(f"False: {total_metrics['false_alarm']} ({total_metrics['false_alarm']/total*100:.2f}%)")
    else:
        print("No videos processed.")
    print("-" * 60)
    
    for log in all_results_log:
        print(log)

if __name__ == "__main__":
    main()