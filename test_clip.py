import cv2
import torch
import clip
from PIL import Image
import numpy as np
import sys
import os

# === 配置部分 ===
VIDEO_PATH = "/data2/pyh/video_stream_event_detection/zhongjifangan/dataset/PxTAy6kI9c4_000370_000380.mp4"
MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 使用 config.yaml 中的详细描述，效果比单纯的“逗猫”两个字要好得多
EVENTS = {
    "进门出门": "一个人正在穿过门口",
    "逗猫": "一个人正在与猫互动玩耍",
    "聚餐": "人们围坐在餐桌旁一起用餐",
    "跌倒": "一个人跌倒在地上"
}

def main():
    print(f"🔧 Using device: {DEVICE}")
    
    # 1. 检查视频文件
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found at {VIDEO_PATH}")
        return

    # 2. 加载 CLIP 模型
    print(f"📦 Loading CLIP model: {MODEL_NAME}...")
    try:
        model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
        print("✅ CLIP loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load CLIP: {e}")
        print("提示: 请确保已安装 openai-clip: pip install git+https://github.com/openai/CLIP.git")
        return

    # 3. 预计算文本特征 (Text Embeddings)
    print("📝 Encoding text descriptions...")
    text_features_dict = {}
    
    model.eval()
    with torch.no_grad():
        for name, desc in EVENTS.items():
            # Tokenize
            text_inputs = clip.tokenize([desc]).to(DEVICE)
            # Encode
            text_feat = model.encode_text(text_inputs)
            # 归一化 (关键步骤，否则计算出的相似度数值不对)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_features_dict[name] = text_feat
    
    print(f"✅ Encoded {len(text_features_dict)} events.")

    # 4. 处理视频帧
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"▶️  Processing video: {VIDEO_PATH}")
    print(f"   Total frames: {total_frames}, FPS: {fps}")
    print("-" * 60)
    print(f"{'Frame':<8} | {'Time(s)':<8} | {'进门出门':<10} | {'逗猫':<10} | {'聚餐':<10} | {'跌倒':<10}")
    print("-" * 60)

    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # === 核心修复步骤：BGR 转 RGB ===
            # OpenCV 读入是 BGR，CLIP 需要 RGB。如果不转，分数会极低或错乱。
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转为 PIL Image 并预处理
            pil_image = Image.fromarray(frame_rgb)
            image_input = preprocess(pil_image).unsqueeze(0).to(DEVICE)
            
            # 计算图像特征
            with torch.no_grad():
                image_feat = model.encode_image(image_input)
                # 归一化
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                
                # 计算相似度 (Image @ Text.T)
                # 结果是一个字典，存储每个事件的分数
                scores = {}
                for name, text_feat in text_features_dict.items():
                    # 矩阵乘法计算余弦相似度
                    similarity = (image_feat @ text_feat.T).item()
                    scores[name] = similarity
            
            # 打印结果
            frame_time = frame_idx / fps if fps > 0 else 0
            print(f"{frame_idx:<8} | {frame_time:<8.2f} | "
                  f"{scores['进门出门']:<10.4f} | "
                  f"{scores['逗猫']:<10.4f} | "
                  f"{scores['聚餐']:<10.4f} | "
                  f"{scores['跌倒']:<10.4f}")
            
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        cap.release()
        print("-" * 60)
        print("✅ Done.")

if __name__ == "__main__":
    main()