import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import os

def run_hierarchical_clustering(embeddings: np.ndarray, method: str = "ward", metric: str = "euclidean") -> np.ndarray:
    """
    階層的クラスタリングを実行し、リンク行列を返します。
    """
    return linkage(embeddings, method=method, metric=metric)

def get_flat_clusters(linkage_matrix: np.ndarray, num_clusters: int, criterion: str = "maxclust") -> np.ndarray:
    """
    リンク行列から指定されたクラスタ数で平坦なクラスタラベルを取得します。
    """
    return fcluster(linkage_matrix, t=num_clusters, criterion=criterion)

def save_cluster_labels(cluster_labels: np.ndarray, output_dir: str, file_prefix: str, num_clusters: int):
    """
    クラスタラベルを.npyファイルに保存します。
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{file_prefix}_{num_clusters}.npy")
    np.save(filename, cluster_labels)
    print(f'クラスタリング結果を {filename} に保存しました。')
