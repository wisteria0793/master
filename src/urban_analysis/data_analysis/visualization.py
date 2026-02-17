import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import os
from typing import Optional, List

def plot_dendrogram(linkage_matrix, names: Optional[List[str]] = None, output_path: Optional[str] = None):
    """
    デンドログラムを描画し、オプションでファイルに保存します。
    """
    plt.figure(figsize=(15, 10))
    dendrogram(linkage_matrix,
                orientation='top',
                labels=names,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    if output_path:
        # ディレクトリがない場合は作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"デンドログラムを {output_path} に保存しました。")
        plt.close()
    else:
        plt.show()
