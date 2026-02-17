import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from urban_analysis.data_analysis.metrics import calculate_silhouette_score
from urban_analysis.config import PROCESSED_DATA_DIR

def load_labels(file_path: Path, column_name: str = "cluster") -> np.ndarray:
    """ラベルファイルを読み込む (.npy or .csv)"""
    if file_path.suffix == ".npy":
        return np.load(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        if column_name in df.columns:
            return df[column_name].values
        else:
            print(f"警告: CSVに列 '{column_name}' が見つかりません。最初の列を使用します。")
            return df.iloc[:, 0].values
    else:
        raise ValueError(f"サポートされていないラベル形式です: {file_path.suffix}")

def load_embeddings(file_path: Path) -> np.ndarray:
    """埋め込みファイルを読み込む (.npy or .csv)"""
    if file_path.suffix == ".npy":
        return np.load(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        # 数値列のみ抽出
        return df.select_dtypes(include=[np.number]).values
    else:
        raise ValueError(f"サポートされていない埋め込み形式です: {file_path.suffix}")

def main():
    parser = argparse.ArgumentParser(description="埋め込みとクラスタラベルからシルエットスコアを計算します。")
    parser.add_argument("--embeddings", type=str, required=True, help="埋め込みベクトルファイル (.npy or .csv)")
    parser.add_argument("--labels", type=str, required=True, help="クラスタラベルファイル (.npy or .csv)")
    parser.add_argument("--label_col", type=str, default="cluster", help="CSVの場合のラベル列名 (デフォルト: cluster)")
    
    args = parser.parse_args()
    
    embeddings_path = Path(args.embeddings)
    labels_path = Path(args.labels)
    
    if not embeddings_path.exists():
        print(f"エラー: 埋め込みファイルが見つかりません: {embeddings_path}")
        return
        
    if not labels_path.exists():
        print(f"エラー: ラベルファイルが見つかりません: {labels_path}")
        return
        
    try:
        print(f"埋め込みを読み込み中: {embeddings_path}")
        embeddings = load_embeddings(embeddings_path)
        
        print(f"ラベルを読み込み中: {labels_path}")
        labels = load_labels(labels_path, args.label_col)
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました: {e}")
        return
    
    if len(embeddings) != len(labels):
        print(f"エラー: 埋め込みの数 ({len(embeddings)}) とラベルの数 ({len(labels)}) が一致しません。")
        return
        
    print(f"データ数: {len(embeddings)}, 埋め込み次元: {embeddings.shape[1]}")
    print("シルエットスコアを計算中...")
    score = calculate_silhouette_score(embeddings, labels)
    
    print(f"\n結果:")
    print(f"シルエットスコア: {score:.4f}")
    print("\n※スコアは -1 から 1 の範囲で、1に近いほどクラスタが適切に分離されていることを示します。")

if __name__ == "__main__":
    main()
