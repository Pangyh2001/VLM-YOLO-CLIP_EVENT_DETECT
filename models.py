import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from ultralytics import YOLOWorld
try:
    import clip
    USE_OPENAI_CLIP = True
except ImportError:
    import open_clip
    USE_OPENAI_CLIP = False
    print("⚠️  OpenAI CLIP not found, using open_clip instead")

from typing import List, Tuple, Dict, Any
import os

class ModelManager:
    """管理所有模型的加载和推理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        # 初始化模型
        self._init_yolo()
        self._init_clip()
        self._init_vlm()
        
    def _init_yolo(self):
        """初始化 YOLO-World 模型"""
        print("📦 Loading YOLO-World...")
        model_name = self.config['models']['yolo_model']
        
        # 确保模型目录存在
        os.makedirs("./models", exist_ok=True)
        
        # YOLO-World 会自动下载到 ./models/
        self.yolo = YOLOWorld(model_name)
        print("✅ YOLO-World loaded")
        
    def _init_clip(self):
        """初始化 CLIP 模型"""
        print("📦 Loading CLIP...")
        model_name = self.config['models']['clip_model']
        
        if USE_OPENAI_CLIP:
            # 使用 OpenAI 的 CLIP
            # 模型名称格式转换: openai/clip-vit-base-patch32 -> ViT-B/32
            model_name_lower = model_name.lower()
            
            if 'vit-base-patch32' in model_name_lower or 'vit-b-32' in model_name_lower:
                clip_name = 'ViT-B/32'
            elif 'vit-base-patch16' in model_name_lower or 'vit-b-16' in model_name_lower:
                clip_name = 'ViT-B/16'
            elif 'vit-large-patch14-336' in model_name_lower or 'vit-l-14-336' in model_name_lower:
                clip_name = 'ViT-L/14@336px'
            elif 'vit-large-patch14' in model_name_lower or 'vit-l-14' in model_name_lower:
                clip_name = 'ViT-L/14'
            elif 'rn50x64' in model_name_lower:
                clip_name = 'RN50x64'
            elif 'rn50x16' in model_name_lower:
                clip_name = 'RN50x16'
            elif 'rn50x4' in model_name_lower:
                clip_name = 'RN50x4'
            elif 'rn101' in model_name_lower:
                clip_name = 'RN101'
            elif 'rn50' in model_name_lower:
                clip_name = 'RN50'
            else:
                clip_name = 'ViT-B/32'  # 默认
            
            print(f"   Loading CLIP model: {clip_name}")
            self.clip_model, self.clip_preprocess = clip.load(clip_name, device=self.device)
        else:
            # 使用 open_clip
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
            )
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        print("✅ CLIP loaded")
        
    def _init_vlm(self):
        """初始化 VLM (Qwen2.5-VL) 模型"""
        print("📦 Loading Qwen2.5-VL...")
        model_path = self.config['models']['vlm_path']
        
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16，Qwen2.5-VL 推荐
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Qwen2.5-VL loaded")
        
    def set_yolo_classes(self, entities: List[str]):
        """设置 YOLO 要检测的类别"""
        self.yolo.set_classes(entities)
        
    def detect_objects(self, frame: np.ndarray, conf: float = 0.25) -> List[Dict]:
        """
        使用 YOLO 检测物体
        返回: [{'class': str, 'bbox': [x1,y1,x2,y2], 'conf': float, 'id': int}, ...]
        """
        results = self.yolo.track(frame, conf=conf, persist=True, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # 获取跟踪 ID（如果有）
                ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
                
                for i in range(len(boxes)):
                    det = {
                        'class': self.yolo.names[int(classes[i])],
                        'bbox': boxes[i].tolist(),
                        'conf': float(confs[i]),
                        'id': int(ids[i]) if ids is not None else -1
                    }
                    detections.append(det)
                    
        return detections
    
    def compute_clip_similarity(self, image: np.ndarray, text: str) -> float:
        """
        计算图像与文本的 CLIP 相似度
        """
        # 转换为 PIL Image
        pil_image = Image.fromarray(image)
        
        # 预处理
        image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # 计算特征
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            
            if USE_OPENAI_CLIP:
                text_input = clip.tokenize([text]).to(self.device)
                text_features = self.clip_model.encode_text(text_input)
            else:
                text_input = self.clip_tokenizer([text]).to(self.device)
                text_features = self.clip_model.encode_text(text_input)
            
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (image_features @ text_features.T).item()
            
        return similarity
    
    def vlm_verify_event(self, images: List[np.ndarray], event_name: str, 
                         positive_desc: str) -> Tuple[bool, str]:
        """
        使用 VLM 验证事件是否真实发生
        返回: (是否发生, 推理结果文本)
        """
        # 准备提示词
        prompt = f"""请仔细观察这些图像，判断是否正在发生"{event_name}"事件。

事件描述: {positive_desc}

请回答:
1. 这些图像中是否正在发生上述事件？(是/否)
2. 你的判断依据是什么？

请直接以"是"或"否"开头回答。"""

        try:
            # 准备输入
            pil_images = [Image.fromarray(img) for img in images]
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in pil_images],
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 准备输入
            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.vlm_processor(
                text=[text],
                images=pil_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # 生成回答
            with torch.no_grad():
                output_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            # 解码
            generated_text = self.vlm_processor.batch_decode(
                output_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # 提取回答部分（去掉输入的 prompt）
            answer = generated_text.split("assistant\n")[-1].strip()
            
            # 判断是否确认事件
            is_event = answer.startswith("是")
            
            return is_event, answer
            
        except Exception as e:
            print(f"❌ VLM verification error: {e}")
            return False, f"Error: {str(e)}"