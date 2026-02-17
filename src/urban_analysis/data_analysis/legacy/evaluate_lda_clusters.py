# -*- coding: utf-8 -*-
"""
このスクリプトは、LDA（Latent Dirichlet Allocation）によるトピックモデリングの結果（クラスタリング結果）を評価するためのものです。

評価は、以下の2つの異なる特徴空間におけるシルエットスコアを計算することで行います。
1.  **埋め込み表現ベースの評価**:
    各施設の意味的な近さ（埋め込みベクトル）を用いて、LDAクラスタがどれだけ密にまとまっているかを評価します。
    距離尺度にはコサイン類似度が用いられます。

2.  **機能ベースの評価**:
    各施設が持つカテゴリ情報（例: 'restaurant', 'park'など）をMulti-hotベクトル化し、
    機能的な類似性に基づいてクラスタの妥当性を評価します。
    距離尺度にはジャカード距離が用いられます。

■ 処理の流れ
1.  施設情報（JSON）、埋め込みベクトル（Numpy）、LDAのトピック分類結果（CSV）を読み込みます。
2.  施設名をキーとして、3つのデータソースを整列（アライメント）させます。
3.  上記2つの観点（埋め込み、機能）で、それぞれクラスタ全体のシルエットスコアと、クラスタごとの平均シルエットスコアを計算します。
4.  すべての評価結果を一つのCSVファイルに出力します。
"""


import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import MultiLabelBinarizer

# --- 定数定義: ファイルパス ---
EMBEDDINGS_PATH = './data/processed/embedding/sentence-transformer/facility_embeddings.npy'
LDA_RESULTS_PATH = './data/processed//lda/lda_results_cl_10.csv'
FACILITIES_JSON_PATH = './data/processed/poi/filtered_facilities.json'

# Output CSV path
OUTPUT_LDA_EVALUATION_CSV = './data/processed//lda/lda_cluster_evaluation_results_cl_10_categories.csv'

# --- 関数定義 ---
def create_type_vectors_dynamically(facilities_list):
    """
    施設のリストを受け取り、'types'リストの全ての要素を直接使用して、
    Multi-hotベクトルを作成する。
    'types'の全ユニークリストは動的に生成される。

    Args:
        facilities_list (list): 施設情報の辞書のリスト。

    Returns:
        numpy.ndarray: Multi-hotベクトル。
    """
    # Extract types from all facilities, handling missing data
    all_types_lists = [
        # f.get('google_places_data', {}).get('details', {}).get('types', [])
        f.get('categories', [])
        for f in facilities_list
    ]

    # The MultiLabelBinarizer can handle fitting and transforming in one step
    mlb = MultiLabelBinarizer()
    type_vectors = mlb.fit_transform(all_types_lists)
    
    print(f"動的に生成されたカテゴリ（types）の数: {len(mlb.classes_)}")
    # print("検出されたカテゴリ一覧:", mlb.classes_) # Uncomment for debugging

    return type_vectors


# --- メインの実行部分 ---
if __name__ == "__main__":
    all_results = []

    try:
        # --- 1. データの読み込みとアライメント ---
        print("--- 1. データの読み込みとアライメント ---")

        # マスターデータとしてJSONファイルを読み込む (これが順序の基準となる)
        with open(FACILITIES_JSON_PATH, 'r', encoding='utf-8') as f:
            facilities_data = json.load(f)
        
        # 埋め込みとLDA結果を読み込む
        embeddings = np.load(EMBEDDINGS_PATH)
        df_lda = pd.read_csv(LDA_RESULTS_PATH)

        print(f"マスターリストを読み込みました: {len(facilities_data)}件")
        print(f"埋め込みデータを読み込みました。Shape: {embeddings.shape}")
        print(f"LDA結果を読み込みました: {len(df_lda)}件")

        # --- データの整合性チェックとアライメント ---
        if embeddings.shape[0] != len(facilities_data):
            raise ValueError(
                f"埋め込み({embeddings.shape[0]}件)とJSON({len(facilities_data)}件)の施設数が一致しません。"
            )

        # LDA結果を施設名でマッピング(辞書作成)
        df_lda = df_lda.drop_duplicates(subset=['facility_name'])
        topic_map = df_lda.set_index('facility_name')['dominant_topic'].to_dict()

        # JSONファイルの施設名順に、LDAのトピックを並べ替えてラベルを作成
        labels = []
        for facility in facilities_data:
            facility_name = facility.get('name')
            topic = topic_map.get(facility_name, -1) # LDA結果にない施設は-1とする
            labels.append(topic)
        
        labels = np.array(labels)
        
        # LDA結果に存在しなかった施設を除外して再アライメント
        valid_indices = np.where(labels != -1)[0]
        if len(valid_indices) != len(facilities_data):
            print(f"警告: LDA結果に存在しない施設が{len(facilities_data) - len(valid_indices)}件あったため、分析から除外します。")
            labels = labels[valid_indices]
            embeddings = embeddings[valid_indices]
            facilities_data = [facilities_data[i] for i in valid_indices]
        
        print(f"アライメント後の分析対象施設数: {len(labels)}")
        if len(labels) == 0:
            raise ValueError("分析対象の施設が0件です。施設名のマッチングを確認してください。")

        # --- 2. 埋め込み表現に基づくクラスタ評価 ---
        print("\n--- 2. 埋め込み表現に基づくLDAクラスタ評価 ---")
        score_embedding = silhouette_score(embeddings, labels, metric='cosine')
        print(f"\n埋め込みに基づく全体のシルエットスコア (Cosine): {score_embedding:.4f}")
        
        print("\n--- クラスタごとの平均シルエットスコア (埋め込み) ---")
        samples_scores_emb = silhouette_samples(embeddings, labels, metric='cosine')
        df_results_emb = pd.DataFrame({'label': labels, 'score': samples_scores_emb})
        cluster_avg_scores_emb = df_results_emb.groupby('label')['score'].mean()
        print(cluster_avg_scores_emb)

        # Collect embedding evaluation results
        all_results.append(pd.DataFrame({
            'Evaluation Type': 'LDA-Embedding-based',
            'Metric': ['Overall Silhouette Score (Cosine)'] + [f'Cluster {c} Avg Silhouette Score' for c in cluster_avg_scores_emb.index],
            'Value': [score_embedding] + cluster_avg_scores_emb.tolist()
        }))

        # --- 3. 機能（types）に基づくLDAクラスタ評価 ---
        print("\n--- 3. 機能（types）に基づくLDAクラスタ評価 ---")
        # Create type vectors dynamically from all available types
        type_vectors = create_type_vectors_dynamically(facilities_data)

        # Calculate silhouette score based on type vectors
        score_functional = silhouette_score(type_vectors, labels, metric='jaccard')
        print(f"\n機能的タイプに基づく全体のシルエットスコア (Jaccard): {score_functional:.4f}")

        print("\n--- クラスタごとの平均シルエットスコア (機能) ---")
        samples_scores_func = silhouette_samples(type_vectors, labels, metric='jaccard')
        df_results_func = pd.DataFrame({'label': labels, 'score': samples_scores_func})
        cluster_avg_scores_func = df_results_func.groupby('label')['score'].mean()
        print(cluster_avg_scores_func)

        # Collect functional evaluation results
        all_results.append(pd.DataFrame({
            'Evaluation Type': 'LDA-Functional-based',
            'Metric': ['Overall Silhouette Score (Jaccard)'] + [f'Cluster {c} Avg Silhouette Score' for c in cluster_avg_scores_func.index],
            'Value': [score_functional] + cluster_avg_scores_func.tolist()
        }))

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。パスを確認してください.\n{e}")
    except (KeyError, ValueError) as e:
        print(f"エラー: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

    # Save all results to a single CSV
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_LDA_EVALUATION_CSV, index=False, encoding='utf-8-sig')
        print(f"すべてのLDA評価結果を '{OUTPUT_LDA_EVALUATION_CSV}' に保存しました。")