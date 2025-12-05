#!/bin/bash

# 视频流事件检测系统 - 使用示例

echo "=================================="
echo "视频流事件检测系统 - 使用示例"
echo "=================================="
echo ""

# 示例 1: 测试系统
echo "1️⃣  运行系统测试"
echo "   python test_detection.py"
echo ""

# 示例 2: 本地视频文件（基础）
echo "2️⃣  处理本地视频文件"
echo "   python run_detection.py --source video.mp4"
echo ""

# 示例 3: 本地视频文件 + 实时显示
echo "3️⃣  处理本地视频 + 实时显示"
echo "   python run_detection.py --source video.mp4 --display"
echo ""

# 示例 4: RTSP 流
echo "4️⃣  处理 RTSP 视频流"
echo "   python run_detection.py --source rtsp://192.168.1.100:554/stream1"
echo ""

# 示例 5: AsyncAVLoader（实时模式）
echo "5️⃣  使用 AsyncAVLoader - 实时模式"
echo "   python run_detection.py --source async \\"
echo "       --async-videos video1.mp4 video2.mp4 \\"
echo "       --async-mode realtime \\"
echo "       --async-speed 1.0"
echo ""

# 示例 6: AsyncAVLoader（快速模式）
echo "6️⃣  使用 AsyncAVLoader - 快速模式"
echo "   python run_detection.py --source async \\"
echo "       --async-videos video.mp4 \\"
echo "       --async-mode fast"
echo ""

# 示例 7: 自定义配置
echo "7️⃣  使用自定义配置文件"
echo "   python run_detection.py --source video.mp4 --config my_config.yaml"
echo ""

echo "=================================="
echo "提示:"
echo "- 按 Ctrl+C 可以随时停止处理"
echo "- 使用 --display 可以实时查看检测结果"
echo "- 结果会自动保存到 detection_results.json"
echo "=================================="