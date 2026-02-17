# -*- coding: utf-8 -*-
"""
保存されているすべてのストリートビュー画像に対して、
StreetCLIPモデルを用いて画像埋め込み表現（ベクトル）を生成し、CSVおよびNPY形式で保存するスクリプト。

入力: data/raw/street_view_images_50m_optimized/*.jpg
出力: 
    - data/processed/streetclip_embeddings/streetclip_features.csv (全データ)
    - data/processed/streetclip_embeddings/streetclip_embeddings.npy (ベクトルのみ)
    - data/processed/streetclip_embeddings/streetclip_metadata.csv (メタデータのみ)
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from tqdm import tqdm
import re
import sys
from pathlib import Path

# プロジェクトルートをパスに追加して config をインポート可能にする
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.urban_analysis.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# --- 設定 ---
IMAGE_DIR = RAW_DATA_DIR / 'street_view_images_50m_optimized'
OUTPUT_DIR = PROCESSED_DATA_DIR / 'streetclip_embeddings'
OUTPUT_CSV_FILE = OUTPUT_DIR / 'streetclip_features.csv'
OUTPUT_NPY_FILE = OUTPUT_DIR / 'streetclip_embeddings.npy'
OUTPUT_META_FILE = OUTPUT_DIR / 'streetclip_metadata.csv'

MODEL_NAME = "geolocal/StreetCLIP"
BATCH_SIZE = 32  # メモリに応じて調整してください

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    # Pathオブジェクトからglobする場合
    image_paths = list(IMAGE_DIR.glob("*.jpg"))
    # 文字列のリストに変換
    image_paths = [str(p) for p in image_paths]
    
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
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return

    model.to(device)
    model.eval()

    all_features = []
    metadata = []
    
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
                    all_features.append(feat)
                    metadata.append({
                        'point_id': point_id,
                        'angle': angle,
                        'direction': direction,
                        'filename': os.path.basename(path)
                    })
                    
        except Exception as e:
            print(f"バッチ処理エラー: {e}")
            continue

    if not all_features:
        print("特徴量が生成されませんでした。")
        return

    # NumPy配列に変換
    all_features_np = np.array(all_features)
    print(f"埋め込みベクトルの形状: {all_features_np.shape}")

    # DataFrame作成 (ソート用)
    df_meta = pd.DataFrame(metadata)
    
    # メタデータと特徴量を結合してソートする（一貫性のため）
    # インデックスを保持してソート
    df_meta['original_index'] = range(len(df_meta))
    df_meta.sort_values(by=['point_id', 'angle'], inplace=True)
    sorted_indices = df_meta['original_index'].values
    
    # ソートされた順序で特徴量とメタデータを再構築
    df_meta_sorted = df_meta.drop(columns=['original_index']).reset_index(drop=True)
    all_features_sorted = all_features_np[sorted_indices]

    # 保存 1: NPY (ベクトルのみ)
    print(f"NPYファイルに保存中: {OUTPUT_NPY_FILE}")
    np.save(OUTPUT_NPY_FILE, all_features_sorted)

    # 保存 2: CSV (メタデータのみ)
    print(f"メタデータCSVに保存中: {OUTPUT_META_FILE}")
    df_meta_sorted.to_csv(OUTPUT_META_FILE, index=False)

    # 保存 3: CSV (結合データ - 既存互換性のため)
    print(f"結合CSVを作成中...")
    # 特徴量をDataFrame化
    feature_cols = [f'feat_{k}' for k in range(all_features_sorted.shape[1])]
    df_features = pd.DataFrame(all_features_sorted, columns=feature_cols)
    
    df_combined = pd.concat([df_meta_sorted, df_features], axis=1)
    
    print(f"結合CSVに保存中: {OUTPUT_CSV_FILE}")
    df_combined.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print("完了しました。")

if __name__ == '__main__':
    main()
