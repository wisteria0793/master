from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import os

def load_streetclip_model():
    """StreetCLIPモデルとプロセッサをロードする"""
    processor = AutoProcessor.from_pretrained("geolocal/StreetCLIP")
    model = AutoModelForZeroShotImageClassification.from_pretrained("geolocal/StreetCLIP")
    return processor, model

def load_blip_model():
    """BLIPモデルとプロセッサをロードする"""
    caption_model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(caption_model_id)
    model = BlipForConditionalGeneration.from_pretrained(caption_model_id)
    return processor, model

def classify_with_streetclip(image, candidate_labels, processor, model):
    """StreetCLIPを使用して画像を分類する"""
    inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    scores = probs.squeeze().tolist()
    
    results = sorted(zip(candidate_labels, scores), key=lambda x: x[1], reverse=True)
    return results

def extract_image_features(image, processor, model):
    """StreetCLIPを使用して画像ベクトルを抽出する"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs.pixel_values)
    return image_features

def generate_caption_with_blip(image, processor, model):
    """BLIPを使用して画像キャプションを生成する"""
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def main():
    # --- 1. 画像の準備 ---
    image_path = "data/raw/street_view_images_50m_optimized/pano_Z_vLZBn614K2XIsJUfla1g_h270.jpg"
    
    if not os.path.exists(image_path):
        print(f"エラー: '{image_path}' が見つかりませんでした。")
        return

    image = Image.open(image_path).convert("RGB")

    # --- 2. StreetCLIP による処理 ---
    print("StreetCLIPモデルをロード中...")
    sc_processor, sc_model = load_streetclip_model()
    
    # 分類
    candidate_labels = [
        "concrete road",
        "cobblestone road",
        "brick road",
        "gravel road",
        "dirt road",
        "grassy trail"
    ]
    
    # print("--- StreetCLIP 分類結果 ---")
    # classification_results = classify_with_streetclip(image, candidate_labels, sc_processor, sc_model)
    # for label, score in classification_results:
    #     print(f"{label}: {score:.4f}")

    # # 特徴抽出
    # features = extract_image_features(image, sc_processor, sc_model)
    # print(f"\n抽出された画像ベクトルの形状: {features.shape}")

    # --- 3. BLIP によるキャプション生成 ---
    print("\nBLIPモデルをロード中...")
    blip_processor, blip_model = load_blip_model()
    
    print("--- 画像キャプション生成 (BLIP) ---")
    caption = generate_caption_with_blip(image, blip_processor, blip_model)
    print(f"生成されたキャプション: {caption}")

if __name__ == "__main__":
    main()
