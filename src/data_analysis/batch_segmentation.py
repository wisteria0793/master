"""
batch_segmentation.py

このスクリプトは、指定された入力ディレクトリ内の複数の画像に対して、セマンティックセグメンテーションをバッチ処理で実行します。
各画像はCityscapesデータセットで学習済みのSegFormerモデル（nvidia/segformer-b5-finetuned-cityscapes-1024-1024）を使用して分析され、以下の結果を生成します。

機能:
1.  **セマンティックセグメンテーション**: 画像内の各ピクセルを「道路」「建物」「植生」などの定義済みカテゴリーに分類します。
2.  **クラス別占有率の計算**: 各画像について、全セグメンテーションクラス（特に「vegetation」としての緑地率）が画像全体に占める割合をパーセンテージで算出します。
3.  **結果の可視化**: 各画像のセグメンテーション結果を色分けされた画像ファイルとして、指定された出力ディレクトリに保存します。
4.  **結果の集計**: 全画像のクラス別占有率を一つのCSVファイル（segmentation_ratios.csv）にまとめ、出力ディレクトリに保存します。

使用方法:
-   `INPUT_DIRECTORY`: 処理したい画像ファイルが格納されているディレクトリのパスを設定します。
-   `OUTPUT_DIRECTORY`: 結果のセグメンテーション画像とCSVファイルが保存されるディレクトリのパスを設定します。
    (出力ディレクトリはスクリプトが自動で作成します。)

必要なライブラリ:
-   `transformers`
-   `torch`
-   `timm`
-   `matplotlib`
-   `Pillow`
-   `numpy`
-   `pandas`
-   `tqdm`
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import matplotlib.pyplot as plt

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
    except Exception as e:
        print(f"画像を読み込み中にエラーが発生しました ({image_path}): {e}")
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

def save_segmentation_visualization(original_image, raw_ids, model, save_path):
    """
    セグメンテーション結果を可視化してファイルに保存します (表示はしない)。
    """
    id2label = model.config.id2label
    num_classes = len(id2label)
    
    cityscapes_palette = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
        [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
        [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]
    unlabeled_color = [0, 0, 0]
    palette = cityscapes_palette + [unlabeled_color] * (num_classes - len(cityscapes_palette))

    mask_rgb = np.zeros((raw_ids.shape[0], raw_ids.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        mask_rgb[raw_ids == class_id] = color

    # 結果を一枚の画像として保存
    result_image = Image.fromarray(mask_rgb)
    result_image.save(save_path)
    # メモリ解放のためfigureを閉じる
    plt.close('all')

def calculate_ratios(raw_segmentation_ids, model):
    """
    生のセグメンテーションIDテンソルから各クラスの占有率を計算し、辞書として返します。
    """
    id2label = model.config.id2label
    total_pixels = raw_segmentation_ids.numel()
    
    unique_ids, counts = torch.unique(raw_segmentation_ids, return_counts=True)
    
    ratios = {}
    for class_id, count in zip(unique_ids.tolist(), counts.tolist()):
        label_name = id2label.get(class_id, "unknown")
        ratio = (count / total_pixels) * 100
        ratios[label_name] = ratio
        
    return ratios

if __name__ == "__main__":
    # --- ユーザーが編集する部分 ---
    INPUT_DIRECTORY = "/Users/atsuyakatougi/Desktop/master/data/raw/street_view_images_50m_optimized"
    OUTPUT_DIRECTORY = "/Users/atsuyakatougi/Desktop/master/data/processed/segmentation_results_50m"
    # --- 編集はここまで ---

    # 出力ディレクトリを作成
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # 1. モデルのロード (一度だけ)
    processor, model = load_segformer_model()
    if torch.cuda.is_available():
        model.to('cuda')

    # 2. 画像リストの取得
    try:
        image_filenames = [f for f in os.listdir(INPUT_DIRECTORY) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_filenames:
            print(f"エラー: 入力ディレクトリ '{INPUT_DIRECTORY}' に画像が見つかりません。")
            exit()
    except FileNotFoundError:
        print(f"エラー: 入力ディレクトリ '{INPUT_DIRECTORY}' が見つかりません。")
        exit()

    # 3. 各画像に対して処理を実行
    all_results = []
    for filename in tqdm(image_filenames, desc="セグメンテーション処理中"):
        image_path = os.path.join(INPUT_DIRECTORY, filename)
        
        original_image, raw_ids = segment_with_segformer(processor, model, image_path)
        
        if original_image is None:
            continue

        # a. 比率の計算
        ratios = calculate_ratios(raw_ids, model)
        ratios['filename'] = filename
        all_results.append(ratios)

        # b. 可視化結果の保存
        output_image_path = os.path.join(OUTPUT_DIRECTORY, f"{os.path.splitext(filename)[0]}_seg.png")
        save_segmentation_visualization(original_image, raw_ids, model, output_image_path)

    # 4. 全結果をCSVファイルに保存
    if all_results:
        csv_path = os.path.join(OUTPUT_DIRECTORY, "segmentation_ratios.csv")
        df = pd.DataFrame(all_results)
        
        # 'filename'列を先頭に移動
        cols = ['filename'] + [col for col in df.columns if col != 'filename']
        df = df[cols]

        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"\n全画像のクラス別占有率を {csv_path} に保存しました。")

    print("\nすべての処理が完了しました。")
