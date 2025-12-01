import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import os

def main():
    """
    画像埋め込みベクトルをロードし、階層的クラスタリングを実行して、
    結果をデンドログラムとして保存する。
    """
    # --- 1. データのロード ---
    embedding_path = 'data/processed/embedding/clip/image_embeddings.npy'
    output_dir = 'docs/results'
    output_path = os.path.join(output_dir, 'image_embedding_dendrogram.png')

    if not os.path.exists(embedding_path):
        print(f"エラー: 埋め込みファイルが見つかりません: {embedding_path}")
        return

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    print("画像埋め込みをロード中...")
    embeddings = np.load(embedding_path)
    print(f"ロード完了。埋め込みの数: {embeddings.shape[0]}")

    # --- 2. 類似度/距離の計算 ---
    # コサイン類似度は-1から1の値を取る。距離として扱うために 1 - similarity を計算する。
    print("画像間のコサイン距離を計算中...")
    cosine_dist = 1 - cosine_similarity(embeddings)
    print("計算完了。")

    # --- 3. 階層的クラスタリング ---
    # ward法は、各ステップでクラスタ内の分散を最小化するペアを併合する方法。
    print("階層的クラスタリングを実行中（ward法）...")
    linked = linkage(cosine_dist, method='ward')
    print("クラスタリング完了。")

    # --- 4. デンドログラムによる可視化 ---
    print("デンドログラムを生成中...")
    plt.figure(figsize=(20, 12))
    dendrogram(
        linked,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_font_size=8,
    )
    plt.title('Dendrogram of Flickr Image Embeddings (Ward Linkage)', fontsize=16)
    plt.xlabel('Image Index', fontsize=12)
    plt.ylabel('Distance (1 - Cosine Similarity)', fontsize=12)
    plt.grid(axis='y')

    # --- 5. 結果の保存 ---
    print(f"デンドログラムを保存中: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n処理が完了しました。")
    print(f"結果は {output_path} に保存されました。")

if __name__ == '__main__':
    main()
