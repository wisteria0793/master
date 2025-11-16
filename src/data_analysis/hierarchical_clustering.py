import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import json
import argparse
import os
import pandas as pd
import re


"""
階層的クラスタリングを実行し、クラスタ数を指定して結果を保存するスクリプト。
このスクリプトは、以下の機能を提供します：
1. 埋め込みベクトルの読み込み
2. 施設情報の読み込み
3. 階層的クラスタリングの実行
4. デンドログラムの描画
5. クラスタ分割と結果の保存
6. 地区ごとの施設数の集計と割合の計算
7. 地区ごとのクラスタ割合をCSVファイルに保存
"""
# 階層的クラスタリング
def hierarchical_clustering(embeddings, method="ward", metric="euclidean"):
    """
    階層的クラスタリングを実行します。
    Args:
        embeddings (np.ndarray): クラスタリングする埋め込みベクトル。
        method (str): リンク方法 (例: "ward", "complete", "average")。
        metric (str): 距離計算のメトリック (例: "euclidean", "cosine")。
    Returns:
        numpy.ndarray: リンク行列。
    """
    return linkage(embeddings, method=method, metric=metric)

# デンドログラムの描画
def plot_dendrogram(linked, names=None, output_path=None):
    """
    デンドログラムを描画し、オプションでファイルに保存します。
    Args:
        linked (numpy.ndarray): linkage関数によって生成されたリンク行列。
        names (list, optional): 各データポイントのラベル。
        output_path (str, optional): デンドログラムを保存するファイルのパス。指定しない場合は表示のみ。
    """
    plt.figure(figsize=(15, 10))
    dendrogram(linked,
                orientation='top',
                labels=names,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram (Ward)')
    plt.xlabel('Sentence Index (or Label)')
    plt.ylabel('Distance')
    
    if output_path:
        plt.savefig(output_path)
        print(f"デンドログラムを {output_path} に保存しました。")
    else:
        plt.show()

# クラスタ分割とファイル保存
def devide_cluster(linked, num_clusters, criterion="maxclust", file_prefix="vector_sentence_bert"):
    """
    階層的クラスタリングの結果を特定のクラスタ数で分割し、結果を.npyファイルに保存します。
    Args:
        linked (numpy.ndarray): linkage関数によって生成されたリンク行列。
        num_clusters (int): 分割するクラスタの数。
        criterion (str): クラスタ分割の基準 (例: "maxclust", "distance")。
        file_prefix (str): 出力ファイル名のプレフィックス。
    Returns:
        numpy.ndarray: 分割されたクラスタラベル。
    """
    devided_clusters = fcluster(linked, t=num_clusters, criterion=criterion)
    output_dir = "hierarchical_clustering"
    os.makedirs(output_dir, exist_ok=True) # ディレクトリが存在しない場合は作成
    output_filename_npy = os.path.join(output_dir, f"{file_prefix}_{num_clusters}.npy")
    np.save(output_filename_npy, devided_clusters)
    print(f'クラスタリング結果を {output_filename_npy} に保存しました。')
    return devided_clusters

# 地区を抽出する関数
def extract_district(address):
    if not isinstance(address, str): return None
    # 「函館市」の後に続く町名（数字やハイフン、長音記号を含まない部分）を抽出
    match = re.search(r'函館市([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+?)(?:[0-9\-－‐‑‒–—―−⸻﹣－]|丁目|番地|町|村|大字|字|$)', address)
    if match:
        district = match.group(1).strip()
        # 特定の例外処理（例: 函館山）
        if '函館山' in district: return '函館山'
        return district
    return None

def main():
    parser = argparse.ArgumentParser(description="階層的クラスタリングを実行し、クラスタ数を指定します。")
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=17, # デフォルトのクラスタ数
        help="階層的クラスタリングで分割するクラスタの数。"
    )
    parser.add_argument(
        "--plot_dendrogram",
        action="store_true",
        help="デンドログラムを表示します。ファイルに保存する場合は --output_dendrogram_path を指定してください。"
    )
    parser.add_argument(
        "--output_dendrogram_path",
        type=str,
        default=None,
        help="デンドログラムを保存するファイルのパス (例: dendrogram.png)。"
    )
    args = parser.parse_args()

    # 埋め込みベクトルと施設情報の読み込み
    try:
        # embeddings_file = './embedding/facility_embeddings_without_categories.npy'
        embeddings_file = 'data/processed/embedding/sentence-transformer/facility_embeddings.npy'
        loaded_embeddings = np.load(embeddings_file)
        print(f"埋め込みベクトルを {embeddings_file} から読み込みました。")
    except FileNotFoundError:
        print(f"エラー: {embeddings_file} が見つかりません。")
        return

    # output_with_google_places_jp.json から施設名と住所を読み込む
    try:
        json_file = 'data/processed/poi/filtered_facilities.json'
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        names = [item.get('name') for item in json_data]
        # 'google_places_data.details.formatted_address' を安全に取得
        addresses = [item.get('google_places_data', {}).get('details', {}).get('formatted_address') for item in json_data]
        
        # 緯度経度も取得
        lats = [item.get('google_places_data', {}).get('details', {}).get('geometry', {}).get('location', {}).get('lat') for item in json_data]
        lngs = [item.get('google_places_data', {}).get('details', {}).get('geometry', {}).get('location', {}).get('lng') for item in json_data]

        df_json = pd.DataFrame({
            'name': names,
            'address': addresses,
            'latitude': lats,
            'longitude': lngs
        })

        print(f"施設情報を {json_file} から読み込みました。")

    except FileNotFoundError:
        print(f"エラー: {json_file} が見つかりません。")
        return
    except json.JSONDecodeError:
        print(f"エラー: {json_file} のJSON形式が不正です。")
        return

    # 階層的クラスタリングの実行
    print("階層的クラスタリングを実行中...")
    linked = hierarchical_clustering(loaded_embeddings, method='ward', metric='euclidean')
    print("階層的クラスタリングが完了しました。")

    # デンドログラムの描画 (オプション)
    if args.plot_dendrogram or args.output_dendrogram_path:
        plot_dendrogram(linked=linked, names=names, output_path=args.output_dendrogram_path)

    # クラスタ分割と保存
    print(f"{args.num_clusters} 個のクラスタに分割します。")
    devided_clusters = devide_cluster(linked, num_clusters=args.num_clusters)

    # クラスタごとの施設名を表示
    if names and devided_clusters is not None:
        print("\n--- クラスタごとの施設 ---")
        clustering_result = [[] for _ in range(args.num_clusters)]
        for i, cluster_label in enumerate(devided_clusters):
            if 0 <= cluster_label - 1 < args.num_clusters:
                clustering_result[cluster_label - 1].append(names[i])

        for i, facilities in enumerate(clustering_result):
            print(f"\n## クラスタ {i+1} ({len(facilities)} 施設)")
            for facility in facilities:
                print(f"- {facility}")
    else:
        print("施設名またはクラスタリング結果が利用できないため、クラスタごとの施設を表示できません。")

    # --- locations_with_clusters_hc_X_with_address.csv の生成 ---
    # クラスタラベルをDataFrameに変換
    df_clusters = pd.DataFrame(devided_clusters, columns=['cluster'])

    # JSONデータとクラスタラベルを結合
    # 結合前に両者の行数が一致しているか確認
    if len(df_json) != len(df_clusters):
        print("エラー: JSONデータとクラスタラベルの行数が一致しません。")
        return

    merged_df = pd.concat([df_json.reset_index(drop=True), df_clusters.reset_index(drop=True)], axis=1)

    # 住所文字列を「函館市」から始まるようにフォーマットする
    def format_hakodate_address(address):
        if not isinstance(address, str):
            return address
        hako_index = address.find("函館市")
        if hako_index != -1:
            return address[hako_index:] # 「函館市」が見つかった場合、そこから後ろを返す
        else:
            return f"函館市{address}" # 見つからない場合、先頭に付与する

    if 'address' in merged_df.columns:
        merged_df['address'] = merged_df['address'].apply(format_hakodate_address)

    # 必要な列を選択し、リネーム
    required_cols = ['name', 'latitude', 'longitude', 'address', 'cluster']
    if all(col in merged_df.columns for col in required_cols):
        df_locations_clusters = merged_df[required_cols]
        
        # 出力ディレクトリの作成
        output_dir = "hierarchical_clustering"
        os.makedirs(output_dir, exist_ok=True)

        # CSVとして保存
        # output_filename_locations = os.path.join(output_dir, f'locations_with_clusters_hc_{args.num_clusters}_with_address_without_categories.csv')
        output_filename_locations = os.path.join(output_dir, f'locations_with_clusters_hc_{args.num_clusters}_with_address.csv')
        df_locations_clusters.to_csv(output_filename_locations, index=False, encoding='utf-8-sig')
        print(f"\nSuccessfully created {output_filename_locations}")
    else:
        print("エラー: locations_with_clusters_hc_X_with_address.csv の生成に必要な列が不足しています。")

    # --- district_cluster_proportions_formatted_X.csv の生成 ---
    if 'address' in df_locations_clusters.columns and 'cluster' in df_locations_clusters.columns:
        df_locations_clusters['district'] = df_locations_clusters['address'].apply(extract_district)
        df_locations_clusters.dropna(subset=['district'], inplace=True)

        # 地区ごとの施設数をカウント
        district_counts = df_locations_clusters.groupby('district').size()

        # 施設数が0以上の地区を抽出 (すべての地区を含める)
        districts_to_keep = district_counts[district_counts >= 0].index

        # 元のDataFrameをフィルタリング
        df_filtered = df_locations_clusters[df_locations_clusters['district'].isin(districts_to_keep)]

        # 地区ごと、クラスタごとのデータ数をカウント
        cluster_counts_filtered = df_filtered.groupby(['district', 'cluster']).size().unstack(fill_value=0)

        # 地区ごとの合計データ数を計算
        district_totals_filtered = cluster_counts_filtered.sum(axis=1)

        # 地区ごとの各クラスタの割合を計算
        cluster_proportions_filtered = cluster_counts_filtered.div(district_totals_filtered, axis=0)

        # すべての数値を小数点以下3桁の文字列としてフォーマット
        cluster_proportions_formatted = cluster_proportions_filtered.applymap(lambda x: f"{x:.3f}")

        # CSVファイルとして保存
        # output_filename_proportions = os.path.join(output_dir, f'district_cluster_proportions_formatted_{args.num_clusters}_without_categories.csv')
        output_filename_proportions = os.path.join(output_dir, f'district_cluster_proportions_formatted_{args.num_clusters}.csv')
        cluster_proportions_formatted.to_csv(output_filename_proportions, encoding='utf-8-sig')
        print(f"Successfully created {output_filename_proportions}")
    else:
        print("エラー: district_cluster_proportions_formatted_X.csv の生成に必要な列が不足しています。")

if __name__ == "__main__":
    main()
