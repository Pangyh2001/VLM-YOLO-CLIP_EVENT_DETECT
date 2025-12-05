#!/usr/bin/env python3
"""
快速测试脚本
用于验证系统是否正常工作
"""

import yaml
import numpy as np
from event_detector import EventDetector


def test_models_loading():
    """测试模型加载"""
    print("=" * 60)
    print("测试 1: 模型加载")
    print("=" * 60)
    
    try:
        config = yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))
        detector = EventDetector(config)
        print("✅ 所有模型加载成功")
        detector.stop()
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False


def test_frame_processing():
    """测试单帧处理"""
    print("\n" + "=" * 60)
    print("测试 2: 单帧处理")
    print("=" * 60)
    
    try:
        config = yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))
        detector = EventDetector(config)
        
        # 创建一个测试帧（纯色图像）
        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        test_frame[:] = (100, 100, 100)  # 灰色背景
        
        print("处理测试帧...")
        result = detector.process_frame(test_frame, 0.0)
        
        print(f"✅ 帧处理成功")
        print(f"   检测到的实体数: {len(result.get('detections', []))}")
        print(f"   事件得分: {result.get('event_scores', {})}")
        print(f"   处理时间: {result.get('processing_time', 0):.3f}s")
        
        detector.stop()
        return True
    except Exception as e:
        print(f"❌ 帧处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试 3: 配置文件")
    print("=" * 60)
    
    try:
        config = yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))
        
        print(f"✅ 配置文件加载成功")
        print(f"   事件数量: {len(config['events'])}")
        print(f"   事件列表:")
        for event in config['events']:
            print(f"      - {event['name']} ({event['type']})")
        
        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False


def main():
    print("\n🧪 开始系统测试...\n")
    
    results = []
    
    # 测试配置文件
    results.append(("配置文件", test_config_loading()))
    
    # 测试模型加载
    results.append(("模型加载", test_models_loading()))
    
    # 测试帧处理
    results.append(("帧处理", test_frame_processing()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！系统运行正常。")
        print("\n现在你可以运行:")
        print("  python run_detection.py --source /path/to/video.mp4")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)