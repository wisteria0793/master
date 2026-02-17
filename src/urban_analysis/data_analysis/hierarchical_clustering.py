import numpy as np
import argparse
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.urban_analysis.data_analysis.clustering import (
    run_hierarchical_clustering, 
    get_flat_clusters
)
from src.urban_analysis.data_analysis.visualization import plot_dendrogram

def perform_clustering(embeddings: np.ndarray, num_clusters: int, method: str = 'ward', metric: str = 'euclidean'):
    """
    ベクトルに対して階層的クラスタリングを適用し、ラベルを返す。
    """
    print(f"階層的クラスタリングを実行中 (method={method}, metric={metric})...")
    linked = run_hierarchical_clustering(embeddings, method=method, metric=metric)
    labels = get_flat_clusters(linked, num_clusters)
    return linked, labels

def main():
    parser = argparse.ArgumentParser(description="ベクトルデータに対して階層的クラスタリングを適用します。")
    parser.add_argument("input", type=str, help="入力となる .npy ファイルのパス")
    parser.add_argument("--num_clusters", type=int, default=10, help="分割するクラスタの数")
    parser.add_argument("--method", type=str, default="ward", help="リンクメソッド (ward, single, complete, averageなど)")
    parser.add_argument("--metric", type=str, default="euclidean", help="距離計算指標")
    parser.add_argument("--output", type=str, default="clusters.npy", help="ラベルの保存先 (.npy)")
    parser.add_argument("--plot", action="store_true", help="デンドログラムを表示")
    
    args = parser.parse_args()

    # データの読み込み
    if not Path(args.input).exists():
        print(f"エラー: ファイルが見つかりません: {args.input}")
        return
    
    embeddings = np.load(args.input)
    print(f"データを読み込みました: {embeddings.shape}")

    # クラスタリングの実行
    linked, labels = perform_clustering(embeddings, args.num_clusters, args.method, args.metric)

    # 結果の保存
    np.save(args.output, labels)
    print(f"クラスタラベルを保存しました: {args.output}")

    # 可視化
    if args.plot:
        plot_dendrogram(linked)

if __name__ == "__main__":
    main()