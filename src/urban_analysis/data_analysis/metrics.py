import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from itertools import combinations
from tqdm.auto import tqdm

def preprocess_segmentation_data(df: pd.DataFrame, group_by_location: bool = True) -> pd.DataFrame:
    """
    セグメンテーション比率データを前処理する。
    - 欠損値を0埋め
    - %表記を0-1確率に変換
    - 必要に応じて地点IDごとに平均化
    """
    # 数値カラムのみ抽出
    numeric_cols = df.select_dtypes(include=np.number).columns
    df_processed = df.copy()
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0) / 100.0

    if group_by_location and 'filename' in df.columns:
        # ファイル名から地点IDを抽出して集約
        # 例: pano_ID_h0.jpg -> pano_ID
        # ファイル名の形式に依存するため、汎用性を高めるには改善が必要
        
        # 簡易的なロジック: '_h' で分割して前半を取る
        df_processed['location_id'] = df_processed['filename'].astype(str).apply(lambda x: x.split('_h')[0] if '_h' in x else x)
        return df_processed.groupby('location_id')[numeric_cols].mean()
    
    return df_processed[numeric_cols]

def calculate_jsd_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレーム（各行が確率分布）間のJSD行列を計算する。
    """
    ids = df.index
    n = len(ids)
    matrix = np.zeros((n, n))
    
    # 行列形式でまとめて計算できれば高速だが、scipyのjensenshannonはペア計算用。
    # Nが大きい場合は並列化を検討すべき。
    
    pairs = list(combinations(range(n), 2))
    values = df.values

    # バリデーション
    if (values < 0).any():
        raise ValueError("確率分布に負の値が含まれています。")

    for i, j in tqdm(pairs, desc="Calculating JSD Matrix"):
        jsd = jensenshannon(values[i], values[j])
        matrix[i, j] = jsd
        matrix[j, i] = jsd
        
    return pd.DataFrame(matrix, index=ids, columns=ids)

def calculate_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    シルエットスコアを計算する。
    
    Args:
        embeddings: 埋め込みベクトルの配列 (n_samples, n_features)
        labels: クラスタラベルの配列 (n_samples,)
        
    Returns:
        float: シルエットスコア
    """
    from sklearn.metrics import silhouette_score
    
    # クラスタ数が2未満、または全サンプルが同一クラスタの場合は計算不可
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("警告: クラスタ数が2未満のため、シルエットスコアを計算できません。")
        return -1.0
        
    score = silhouette_score(embeddings, labels)
    return score
