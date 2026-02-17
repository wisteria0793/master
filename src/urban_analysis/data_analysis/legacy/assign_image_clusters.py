import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster

def assign_clusters_to_images(n_clusters=23):
    """
    画像埋め込みをロードし、階層的クラスタリングを実行後、
    指定された数のクラスタに分類し、各クラスタに属する画像ファイル名を
    JSONファイルに出力する。
    """
    # --- 1. パスの設定 ---
    base_dir = 'data/processed/embedding/clip'
    embedding_path = os.path.join(base_dir, 'image_embeddings.npy')
    filenames_path = os.path.join('data', 'processed', 'images', 'image_filenames.json')


    
    output_dir = 'docs/results'
    output_path = os.path.join(output_dir, f'image_clusters_{n_clusters}.json')

    # --- 2. データのロード ---
    # ファイルの存在チェック
    if not all(os.path.exists(p) for p in [embedding_path, filenames_path]):
        print(f"エラー: 必要なファイルが見つかりません。")
        print(f" - 埋め込み: {embedding_path}")
        print(f" - ファイル名: {filenames_path}")
        return

    print("埋め込みとファイル名をロード中...")
    embeddings = np.load(embedding_path)
    with open(filenames_path, 'r') as f:
        filenames = json.load(f)
    
    if len(embeddings) != len(filenames):
        print("エラー: 埋め込みとファイル名の数が一致しません。")
        return
        
    print(f"ロード完了。{len(filenames)}件のデータ。")

    # --- 3. クラスタリングの実行 ---
    print("画像間のコサイン距離を計算中...")
    cosine_dist = 1 - cosine_similarity(embeddings)

    print("階層的クラスタリングを実行中（ward法）...")
    linked = linkage(cosine_dist, method='ward')

    # --- 4. クラスタへの割り当て ---
    print(f"{n_clusters}個のクラスタに分類中...")
    # 'maxclust'基準で、クラスタ数がn_clustersになるようにカット
    cluster_labels = fcluster(linked, n_clusters, criterion='maxclust')

    # --- 5. 結果の集計 ---
    print("クラスタごとにファイル名を整理中...")
    clusters = {f"cluster_{i}": [] for i in range(1, n_clusters + 1)}
    for i, label in enumerate(cluster_labels):
        clusters[f"cluster_{label}"].append(filenames[i])
    
    # クラスタ番号順にソートするための準備
    sorted_clusters = {f"cluster_{i}": clusters[f"cluster_{i}"] for i in range(1, n_clusters + 1)}

    # --- 6. 結果の保存 ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"結果をJSONファイルに保存中: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_clusters, f, indent=4, ensure_ascii=False)

    print("\n処理が完了しました。")
    print(f"結果は {output_path} に保存されました。")

if __name__ == '__main__':
    # クラスタ数を23に設定して実行
    assign_clusters_to_images(n_clusters=23)
