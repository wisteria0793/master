from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os

def load_hf_model(model_name="facebook/mask2former-swin-large-cityscapes-semantic"):
    """
    Hugging Faceからセグメンテーションモデルとプロセッサをロードします。
    """
    print(f"Hugging Faceモデル '{model_name}' をロード中...")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_name)
    print("モデルのロードが完了しました。")
    return image_processor, model

def segment_with_hf_model(processor, model, image_path):
    """
    Hugging Faceのモデルを使用してセグメンテーションを実行します。
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        return None, None

    # 画像を前処理
    inputs = processor(images=image, return_tensors="pt")

    # GPUが利用可能ならGPUにテンソルを移動
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model.to('cuda')

    # モデルで推論を実行
    with torch.no_grad():
        outputs = model(**inputs)

    # 結果を後処理してセグメンテーションマップを生成
    # target_sizesに元の画像サイズをリストとして渡す
    segmentation_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    
    return image, segmentation_map

def visualize_hf_segmentation(original_image, segmented_map, save_path="hf_segmentation_result.png"):
    """
    元の画像とセグメンテーション結果を並べて表示・保存します。
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_map)
    plt.title("Semantic Segmentation (Cityscapes)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"セグメンテーション結果を {save_path} に保存しました。")
    plt.show()

if __name__ == "__main__":
    # --- ユーザーが編集する部分 ---
    # セグメンテーションを行いたい画像ファイルのフルパスを指定してください
    IMAGE_PATH = "./data/raw/street_view_images_50m_optimized/pano_zvkE2ljaDCUFluC_42mXxA_h90.jpg"
    # --- 編集はここまで ---

    if not os.path.isfile(IMAGE_PATH):
        print(f"エラー: 指定された画像パスにファイルが見つかりません: {IMAGE_PATH}")
    else:
        # 1. モデルのロード
        image_processor, segmentation_model = load_hf_model()
        
        # 2. セグメンテーションの実行
        print(f"\n画像 '{os.path.basename(IMAGE_PATH)}' のセグメンテーションを実行中...")
        original, segmentation_result = segment_with_hf_model(image_processor, segmentation_model, IMAGE_PATH)
        
        # 3. 結果の可視化
        if original is not None:
            print("セグメンテーション結果を可視化中...")
            visualize_hf_segmentation(original, segmentation_result)
            print("\n処理が完了しました。")
