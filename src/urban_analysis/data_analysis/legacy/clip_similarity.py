from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
import os

def calculate_clip_similarity(text_input: str, image_path: str) -> float:
    """
    Calculates the CLIP similarity score between a given text and an image.

    Args:
        text_input (str): The text description.
        image_path (str): The file path to the image.

    Returns:
        float: The cosine similarity score between the text and image embeddings.
               Returns -1.0 if the image file is not found or cannot be opened.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return -1.0

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return -1.0

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    inputs = processor(text=text_input, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    text_features = outputs.text_embeds
    image_features = outputs.image_embeds

    # Normalize embeddings
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = torch.cosine_similarity(text_features, image_features, dim=-1).item()

    return similarity

if __name__ == "__main__":
    # --- ユーザーが編集する部分 ---
    # 比較したい画像ファイルのフルパスを指定してください
    IMAGE_PATH = "data/raw/street_view_images_50m_optimized/pano_-5vChoR95xsBoqiCpXasfQ_h180.jpg"
    
    # 画像と比較するテキストクエリを指定してください
    TEXT_QUERY = "a phot of the blue sky with clouds"
    # --- 編集はここまで ---

    print("CLIPテキスト-画像類似度計算ツール")
    print(f"\n指定された画像: {IMAGE_PATH}")
    print(f"指定されたテキスト: '{TEXT_QUERY}'")

    # 指定されたパスにファイルが存在するかチェック
    if not os.path.isfile(IMAGE_PATH):
        print(f"エラー: 指定されたパスにファイルが見つかりません: {IMAGE_PATH}")
    else:
        print("類似度を計算中...")
        score = calculate_clip_similarity(TEXT_QUERY, IMAGE_PATH)
        
        if score != -1.0:
            print(f"類似度スコア: {score:.4f}")
        else:
            print("類似度の計算に失敗しました。ファイルが破損しているか、サポートされていない形式の可能性があります。")

    print("\nCLIP類似度計算ツールを終了します。")
