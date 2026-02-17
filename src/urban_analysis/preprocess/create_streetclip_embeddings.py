# -*- coding: utf-8 -*-
"""
保存されているすべてのストリートビュー画像に対して、
StreetCLIPモデルを用いて画像埋め込み表現（ベクトル）を生成し、CSVに保存するスクリプト。

入力: data/raw/street_view_images_50m_optimized/*.jpg
出力: data/processed/streetclip_embeddings/streetclip_features.csv
"""

import os
import glob
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from tqdm import tqdm
import re

# --- 設定 ---
BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'streetclip_embeddings')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'streetclip_features.csv')
MODEL_NAME = "geolocal/StreetCLIP"
BATCH_SIZE = 32  # メモリに応じて調整してください

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_filename(filepath):
    """ファイルパスからpoint_idとangleを抽出"""
    filename = os.path.basename(filepath)
    # 想定形式: pano_{point_id}_h{angle}.jpg
    match = re.search(r'pano_(.*)_h(\d+)\.', filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def get_direction_from_angle(angle):
    angle_map = {0: 'front', 90: 'right', 180: 'back', 270: 'left'}
    return angle_map.get(angle, 'unknown')

def main():
    print(f"画像ディレクトリ: {IMAGE_DIR}")
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    print(f"対象画像枚数: {len(image_paths)}")
    
    if len(image_paths) == 0:
        print("画像が見つかりません。パスを確認してください。")
        return

    # デバイス設定 (MacのMPS, CUDA, CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用デバイス: MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用デバイス: CUDA")
    else:
        device = torch.device("cpu")
        print("使用デバイス: CPU")

    print(f"モデル '{MODEL_NAME}' を読み込み中...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    results = []
    
    # バッチ処理
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Embedding生成中"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        batch_images = []
        valid_paths = []
        
        # 画像読み込み
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"画像読み込みエラー: {p}, {e}")
                continue
        
        if not batch_images:
            continue

        # 前処理と推論
        try:
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # get_image_features で埋め込みベクトルを取得
                features = model.get_image_features(**inputs)
                
            # CPUに戻してNumPy化
            features = features.cpu().numpy()
            
            # 結果を格納
            for j, feat in enumerate(features):
                path = valid_paths[j]
                point_id, angle = parse_filename(path)
                direction = get_direction_from_angle(angle)
                
                if point_id:
                    row = {
                        'point_id': point_id,
                        'angle': angle,
                        'direction': direction,
                    }
                    # 特徴量を列として追加 (feat_0, feat_1, ...)
                    for k, val in enumerate(feat):
                        row[f'feat_{k}'] = val
                    
                    results.append(row)
                    
        except Exception as e:
            print(f"バッチ処理エラー: {e}")
            continue

    # DataFrame化して保存
    print("DataFrameを作成中...")
    df = pd.DataFrame(results)
    
    # ソート
    df.sort_values(by=['point_id', 'angle'], inplace=True)
    
    print(f"CSVに保存中: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print("完了しました。")

if __name__ == '__main__':
    main()
