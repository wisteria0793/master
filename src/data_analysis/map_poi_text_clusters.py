import os
import pandas as pd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def get_hex_colors(n):
    """matplotlibのtab20カラーマップからN個のHEXカラーコードを取得する"""
    cmap = plt.get_cmap('tab20')
    return [mcolors.to_hex(cmap(i / n)) for i in range(n)]

def main(n_clusters=8):
    CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', f'hakodate_poi_text_clusters_{n_clusters}.csv')
    OUTPUT_MAP_PATH = os.path.join(BASE_DIR, 'docs', 'results', f'hakodate_poi_text_cluster_map_{n_clusters}.html')

    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} が見つかりません。")
        return

    print("データを読み込み中...")
    df = pd.read_csv(CSV_PATH)
    
    n_clusters_actual = df['text_cluster'].nunique()
    colors = get_hex_colors(max(n_clusters_actual, 20))
    
    # 函館市の中心付近を初期表示位置とする
    center_lat = df['lat'].mean()
    center_lng = df['lng'].mean()
    
    print(f"マップ (k={n_clusters}) を生成中...")
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles='CartoDB positron')
    
    # クラスタごとのレイヤーを作成（表示/非表示を切り替えられるようにする）
    # labels が 1 始まりであることを想定
    feature_groups = {}
    unique_clusters = sorted(df['text_cluster'].unique())
    for cluster_id in unique_clusters:
        fg = folium.FeatureGroup(name=f"Text Cluster {cluster_id}")
        feature_groups[cluster_id] = fg
        m.add_child(fg)

    for _, row in df.iterrows():
        cluster_id = int(row['text_cluster'])
        color = colors[cluster_id % len(colors)]
        
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 200px;">
            <h4 style="margin-bottom: 5px;">{row['name']}</h4>
            <p style="margin: 0;"><b>テキストクラスタ:</b> {cluster_id}</p>
            <p style="margin: 0; font-size: 12px; color: gray;">{row['categories']}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=6,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1
        ).add_to(feature_groups[cluster_id])
    
    # レイヤーコントロールを追加
    folium.LayerControl(collapsed=False).add_to(m)
    
    os.makedirs(os.path.dirname(OUTPUT_MAP_PATH), exist_ok=True)
    m.save(OUTPUT_MAP_PATH)
    print(f"地図上にプロットしたHTMLマップを保存しました: {OUTPUT_MAP_PATH}")

if __name__ == '__main__':
    n_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n_clusters)