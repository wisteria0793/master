from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import logging

class SegmentationModel:
    """セグメンテーションモデルのラッパー"""
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-cityscapes-1024-1024"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Loading segmentation model: {model_name} on {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.id2label = self.model.config.id2label

    def segment(self, image_path_or_obj):
        """画像をセグメンテーションし、クラスIDのマスクを返す"""
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj).convert("RGB")
        else:
            image = image_path_or_obj.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        # 元のサイズにリサイズ
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], 
            mode='bilinear',
            align_corners=False
        )
        
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu()
        return image, pred_seg

def calculate_class_ratios(segmentation_mask, id2label):
    """マスクから各クラスの占有率を計算"""
    total_pixels = segmentation_mask.numel()
    unique_ids, counts = torch.unique(segmentation_mask, return_counts=True)
    
    ratios = {}
    for class_id, count in zip(unique_ids.tolist(), counts.tolist()):
        label_name = id2label.get(class_id, "unknown")
        ratios[label_name] = (count / total_pixels) * 100
        
    return ratios

# Cityscapesのカラーパレット
CITYSCAPES_PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

def create_mask_image(segmentation_mask, palette=CITYSCAPES_PALETTE):
    """クラスIDマスクをRGB画像に変換"""
    mask_np = segmentation_mask.numpy()
    h, w = mask_np.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # パレットが足りない場合の補完
    num_classes = mask_np.max() + 1
    extended_palette = palette + [[0,0,0]] * max(0, num_classes - len(palette))
    
    for class_id, color in enumerate(extended_palette):
        mask_rgb[mask_np == class_id] = color
        
    return Image.fromarray(mask_rgb)
