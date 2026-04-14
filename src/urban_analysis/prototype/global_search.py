import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import get_merged_poi_data

def global_similarity_search(target_name, top_n=800):
    # 1. データの読み込み
    print("全件データを読み込み中...")
    poi_df = get_merged_poi_data()
    
    # 2. 基準となるPOIを特定
    target_rows = poi_df[poi_df['name'].str.contains(target_name, na=False)]
    if target_rows.empty:
        print(f"エラー: 指定されたPOI '{target_name}' が見つかりませんでした。")
        return
    
    target_poi = target_rows.iloc[0]
    target_embedding = np.array(target_poi['text_embedding']).reshape(1, -1)
    
    print(f"\n基準POI: {target_poi['name']}")
    print("-" * 30)
    
    # 3. 全POI（自分以外）との類似度を計算
    other_df = poi_df[poi_df['name'] != target_poi['name']].copy()
    all_embeddings = np.vstack(other_df['text_embedding'].values)
    
    similarities = cosine_similarity(target_embedding, all_embeddings)[0]
    other_df['similarity'] = similarities
    
    # 4. 上位 N 件を抽出
    top_results = other_df.sort_values(by='similarity', ascending=False).head(top_n)
    
    # 5. 結果の表示
    print(f"全 {len(poi_df)} 件の中から『{target_poi['name']}』に内容が似ている上位 {top_n} 件を表示します：\n")
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"{i}. {row['name']}")
        print(f"   - 類似度: {row['similarity']:.4f}")
        # カテゴリやクラスタ情報があれば表示
        if 'category' in row: print(f"   - カテゴリ: {row['category']}")
        print(f"   - クラスタ: {row['cluster']}")
        print("-" * 20)

if __name__ == "__main__":
    import sys
    # コマンドライン引数から検索対象を取得、なければデフォルト
    target = sys.argv[1] if len(sys.argv) > 1 else "五稜郭タワー"
    
    try:
        global_similarity_search(target)
    except Exception as e:
        print(f"実行エラー: {e}")
