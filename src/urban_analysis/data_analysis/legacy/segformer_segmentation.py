from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def load_segformer_model(model_name="nvidia/segformer-b5-finetuned-cityscapes-1024-1024"):
    """
    Hugging FaceからSegFormerモデルとプロセッサをロードします。
    """
    print(f"Hugging Faceモデル '{model_name}' をロード中...")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    print("モデルのロードが完了しました。")
    return image_processor, model

def segment_with_segformer(processor, model, image_path):
    """
    SegFormerモデルを使用してセグメンテーションを実行し、生のクラスIDテンソルを返します。
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

    # Logitsを取得し、元の画像サイズにリサイズ
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1], # (height, width)
        mode='bilinear',
        align_corners=False
    )

    # 各ピクセルのクラスIDを決定
    raw_segmentation_ids = upsampled_logits.argmax(dim=1)[0]
    
    return image, raw_segmentation_ids

def calculate_and_print_ratios(raw_segmentation_ids, model):
    """
    生のセグメンテーションIDテンソルから各クラスの占有率を計算し、表示します。
    特に「緑地率」に注目します。
    """
    print("\n--- クラス別占有率 ---")
    
    id2label = model.config.id2label
    total_pixels = raw_segmentation_ids.numel()
    green_space_ratio = 0.0
    vegetation_class_name = 'vegetation'
    vegetation_id = -1

    for class_id, class_name in id2label.items():
        if class_name == vegetation_class_name:
            vegetation_id = class_id
            break
            
    if vegetation_id == -1:
        print(f"エラー: '{vegetation_class_name}' クラスがモデルのラベルに見つかりません。")
        return

    unique_ids, counts = torch.unique(raw_segmentation_ids, return_counts=True)
    
    print(f"{'クラス名':<15} | {'占有率 (%)'}")
    print(f"{'-'*15} | {'-'*10}")

    for class_id, count in zip(unique_ids.tolist(), counts.tolist()):
        label_name = id2label.get(class_id, "unknown")
        ratio = (count / total_pixels) * 100
        print(f"{label_name:<15} | {ratio:>8.2f}%")
        
        if class_id == vegetation_id:
            green_space_ratio = ratio

    print("\n------------------------")
    print(f"緑地率 (vegetation): {green_space_ratio:.2f}%")
    print("------------------------")

def visualize_segmentation(original_image, raw_ids, model, save_path="segformer_segmentation_result.png"):
    """
    元の画像とセグメンテーション結果を並べて表示・保存します。
    """
    id2label = model.config.id2label
    num_classes = len(id2label)
    
    # クラスごとの色を定義 (Cityscapesの公式カラーマップ)
    cityscapes_palette = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
        [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
        [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]
    
    # 不明なクラス用の色
    unlabeled_color = [0, 0, 0]
    palette = cityscapes_palette + [unlabeled_color] * (num_classes - len(cityscapes_palette))

    # セグメンテーションマスクをRGB画像に変換
    mask_rgb = np.zeros((raw_ids.shape[0], raw_ids.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        mask_rgb[raw_ids == class_id] = color

    # Matplotlibで表示
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_rgb)
    plt.title("Semantic Segmentation (SegFormer/Cityscapes)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nセグメンテーション結果を {save_path} に保存しました。")
    plt.show()

if __name__ == "__main__":
    # --- ユーザーが編集する部分 ---
    # セグメンテーションを行いたい画像ファイルのフルパスを指定してください
    IMAGE_PATH = "./data/raw/hakodate_all_photos_bbox/51373177546_52f07c72f2.jpg"
    # --- 編集はここまで ---

    if not os.path.isfile(IMAGE_PATH):
        print(f"エラー: 指定された画像パスにファイルが見つかりません: {IMAGE_PATH}")
    else:
        # 1. モデルのロード
        processor, model = load_segformer_model()
        
        # 2. セグメンテーションの実行
        print(f"\n画像 '{os.path.basename(IMAGE_PATH)}' のセグメンテーションを実行中...")
        original, raw_ids = segment_with_segformer(processor, model, IMAGE_PATH)
        
        if original is not None:
            # 3. 緑地率などの比率を計算して表示
            calculate_and_print_ratios(raw_ids, model)

            # 4. 結果の可視化
            visualize_segmentation(original, raw_ids, model)
            
            print("\n処理が完了しました。")
