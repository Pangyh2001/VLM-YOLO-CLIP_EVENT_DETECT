# 视频流事件检测系统

这是一个基于 YOLO-World、CLIP 和 Qwen2.5-VL 的智能视频事件检测系统。

## 系统架构

```
视频流 → YOLO检测 → 切图策略 → CLIP初筛 → VLM验证 → 事件记录
         (5/10fps)   (根据事件类型)  (连续5帧)  (异步线程池)  (JSON输出)
```

## 文件结构

```
project/
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖包
├── av_loader.py            # 你提供的视频流加载器
├── models.py               # 模型管理（YOLO/CLIP/VLM）
├── event_processor.py      # 事件处理逻辑（切图、跟踪）
├── vlm_worker.py          # VLM异步验证线程池
├── video_source.py        # 统一视频源接口
├── event_detector.py      # 主检测器
├── run_detection.py       # 主程序
└── models/                # 模型缓存目录（自动创建）
```

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 3. 确保 Qwen2.5-VL 模型路径正确

确认 `/data2/pyh/model_weights/qwen25vl_7b` 存在，或在 `config.yaml` 中修改路径。

## 配置

编辑 `config.yaml` 来配置：

- **模型路径**：VLM、YOLO、CLIP
- **帧率控制**：基础5fps，检测到实体时10fps
- **事件定义**：添加你自己的事件类型
- **阈值参数**：CLIP连续帧数、事件开始/结束帧数
- **输出设置**：JSON保存路径、是否保存图像
- **详细记录**：启用/禁用详细日志，设置采样率

### 详细记录配置

```yaml
output:
  detailed_logging: true        # 启用详细记录
  output_base_dir: "./output"   # 输出目录
  save_yolo_detections: true    # 保存YOLO检测结果
  save_clip_scores: true        # 保存CLIP得分
  save_vlm_results: true        # 保存VLM验证结果
  yolo_sample_rate: 30          # YOLO采样率（每30帧保存一次）
  clip_sample_rate: 10          # CLIP采样率（每10帧保存一次）
```

### 添加新事件

在 `config.yaml` 的 `events` 部分添加：

```yaml
- name: "你的事件名"
  type: "interaction"  # location/interaction/scene/single
  entities:
    - "person"
    - "dog"
  positive_desc: "一个人正在遛狗"
  negative_descs:
    - "一个人和狗在同一空间但没有互动"
    - "人或狗独自待着"
```

## 使用方法

### 1. 本地视频文件

```bash
python run_detection.py --source /path/to/video.mp4
```

### 2. 本地视频 + 实时显示

```bash
python run_detection.py --source /path/to/video.mp4 --display
```

### 3. RTSP 流

```bash
python run_detection.py --source rtsp://192.168.1.100:554/stream
```

### 4. 使用 AsyncAVLoader（模拟流）

```bash
# 实时模式（模拟真实时间流）
python run_detection.py --source async \
    --async-videos video1.mp4 video2.mp4 \
    --async-mode realtime \
    --async-speed 1.0

# 快速模式（加速处理）
python run_detection.py --source async \
    --async-videos video1.mp4 \
    --async-mode fast
```

### 5. 自定义配置文件

```bash
python run_detection.py --source video.mp4 --config my_config.yaml
```

## 输出

### 1. JSON 结果文件

默认保存到 `detection_results.json`：

```json
{
  "metadata": {
    "total_frames": 1500,
    "processed_frames": 750,
    "events_config": [...]
  },
  "results": [
    {
      "event_name": "进门出门",
      "start_time": 10.5,
      "end_time": 12.3,
      "status": "completed",
      "duration": 1.8,
      "vlm_reason": "是，图像中可以看到一个人正在穿过门口..."
    }
  ]
}
```

### 2. 详细记录（如果启用）

当 `detailed_logging: true` 时，会在 `output/` 下创建时间戳文件夹：

```
output/20241206_143025/
├── 1_yolo_detections/          # YOLO检测结果
│   ├── frame_000030_t1.50s.jpg     # 带检测框的图像
│   ├── frame_000030_detections.json # 检测数据
│   ├── frame_000060_t3.00s.jpg
│   └── ...
├── 2_clip_scores/               # CLIP相似度得分
│   ├── clip_scores.csv             # 所有帧的得分数据
│   ├── frame_000010_scores.jpg     # 带得分条形图的图像
│   ├── frame_000020_scores.jpg
│   └── ...
├── 3_vlm_verifications/         # VLM验证结果
│   ├── 进门出门_143056/
│   │   ├── input_frame_1.jpg       # VLM输入帧
│   │   ├── input_frame_2.jpg
│   │   ├── vlm_result_confirmed.jpg # 拼接结果图
│   │   └── vlm_result.json         # 验证详情
│   └── 逗猫_143128/
│       └── ...
└── summary/                     # 汇总报告
    ├── README.md                    # 总览文档
    └── clip_scores_timeline.png     # CLIP得分趋势图
```

#### YOLO检测结果

- **图像**：每30帧保存一次（可配置），带检测框、类别标签、置信度、跟踪ID
- **JSON**：包含所有检测对象的详细信息

#### CLIP相似度得分

- **CSV**：记录每一帧所有事件的相似度得分
- **图像**：每10帧保存一次可视化，显示得分条形图
- **趋势图**：汇总中生成所有事件得分随时间变化的曲线

#### VLM验证结果

- 每次VLM验证创建独立目录
- 保存所有输入帧
- 保存拼接结果图（显示确认/拒绝）
- JSON记录验证详情和推理原因

### 控制台输出

实时显示：
- 检测统计信息（每5秒）
- 事件开始/结束通知
- VLM验证结果
- 帧率变化通知

## 工作原理

### 1. 动态帧率

- **基础模式（5fps）**：未检测到任何实体
- **激活模式（10fps）**：检测到感兴趣的实体
- 连续20帧未检测到 → 降回5fps

### 2. 事件检测流程

```
帧输入 → YOLO检测实体 
       ↓
    有实体? → 否 → 跳过
       ↓ 是
    根据事件类型裁剪图像
       ↓
    CLIP计算所有事件的相似度
       ↓
    某事件连续5帧得分最高?
       ↓ 是
    提交VLM验证队列（异步）
       ↓
    VLM确认? → 是 → 记录事件开始
       ↓
    连续10帧不再最高? → 是 → 记录事件结束
```

### 3. 切图策略

| 事件类型 | 策略 | 示例 |
|---------|------|------|
| location | 原图 + IoU判断 | 进出门 |
| interaction | 所有相关实体的最小外接矩形 | 逗猫 |
| scene | 以主要物体为中心扩展 | 聚餐 |
| single | 单个实体框扩展 | 跌倒 |

### 4. VLM 异步验证

- 3个工作线程并行处理
- 队列大小50
- 非阻塞设计，不影响视频流处理
- 自动负载均衡

## 性能优化建议

1. **GPU 内存**：
   - Qwen2.5-VL 7B 约需 14GB VRAM
   - 可调整 `vlm.max_workers` 来控制并发

2. **处理速度**：
   - 单帧处理：~50-100ms（YOLO+CLIP）
   - VLM验证：~1-3秒/次
   - 动态帧率有效降低计算负载

3. **准确性调优**：
   - 调整 `clip_consecutive_frames`（连续高分帧数）
   - 修改 `positive_desc` 和 `negative_descs` 提高区分度
   - 调整切图扩展比例

## 故障排查

### YOLO 下载失败
```bash
# 手动下载到 ./models/
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-worldv2.pt
```

### CLIP 模型加载失败
```bash
# 需要安装 git-lfs
git lfs install
```

### VLM 内存不足
- 减少 `vlm.max_workers`
- 使用更小的模型（如 Qwen2.5-VL 2B）

### 帧率不稳定
- 检查 `no_detection_frames` 参数
- 确认 GPU 资源充足

## 扩展开发

### 添加新的视频源

在 `video_source.py` 中继承 `VideoSource` 基类：

```python
class MyCustomSource:
    def __iter__(self):
        while True:
            yield {
                'frame': np.array,
                'frame_time': float,
                'frame_idx': int,
                'source': 'custom'
            }
```

### 自定义切图策略

在 `event_processor.py` 的 `crop_image_by_event` 方法中添加新的类型。

### 集成其他 VLM

在 `models.py` 的 `_init_vlm` 和 `vlm_verify_event` 方法中替换模型。

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题，请提交 Issue。