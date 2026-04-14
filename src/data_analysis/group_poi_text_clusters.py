import os
import pandas as pd
import json
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def main(n_clusters=20):
    CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', f'poi_text_clusters_{n_clusters}.csv')
    OUTPUT_MD_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', f'poi_text_clusters_grouped_{n_clusters}.md')
    OUTPUT_JSON_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', f'poi_text_clusters_grouped_{n_clusters}.json')

    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} が見つかりません。")
        return

    df = pd.read_csv(CSV_PATH)
    
    # クラスタ番号で昇順ソート
    df = df.sort_values(by=['text_cluster', 'name'])
    
    grouped = df.groupby('text_cluster')
    
    cluster_dict = {}
    
    with open(OUTPUT_MD_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# POI テキストクラスタ 分類結果 (k={n_clusters})\n\n")
        f.write("Sentence-BERTによる施設紹介文のクラスタリング結果（各クラスタに含まれるPOI一覧）\n\n")
        
        for cluster_id, group in grouped:
            poi_list = group['name'].tolist()
            cluster_dict[int(cluster_id)] = poi_list
            
            f.write(f"## クラスタ {cluster_id} (計 {len(poi_list)} 件)\n")
            for poi in poi_list:
                f.write(f"- {poi}\n")
            f.write("\n")
            
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(cluster_dict, f, ensure_ascii=False, indent=2)
        
    print(f"クラスタごとにグループ化した結果を保存しました:\n - {OUTPUT_MD_PATH}\n - {OUTPUT_JSON_PATH}")

if __name__ == '__main__':
    n_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n_clusters)
