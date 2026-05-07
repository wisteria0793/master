import folium
import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
POI_CLUSTERS_CSV = PROJECT_ROOT / 'data' / 'processed' / 'refined_poi_clusters_k8.csv'
LS_NODES_CSV = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs_individual' / 'landscape' / 'nodes.csv'
LS_EMB_NPY = PROJECT_ROOT / 'data' / 'processed' / 'gnn_results_individual' / 'landscape' / 'embeddings.npy'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# カラーパレット
folium_colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'darkred', 
    'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
    'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black'
]

def get_color(cluster_id, max_colors=17):
    if pd.isna(cluster_id) or cluster_id < 0:
        return 'lightgray'
    return folium_colors[int(cluster_id) % max_colors]

def get_hex_color(cluster_id, colormap):
    # matplotlibなどのカラーマップを使わずに直接指定する簡易版
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    if pd.isna(cluster_id) or cluster_id < 0:
        return '#cccccc'
    cmap = cm.get_cmap(colormap)
    rgba = cmap(int(cluster_id) / 12.0) # 景観は12個想定
    return mcolors.to_hex(rgba)

def main():
    print("データの読み込み中...")
    
    # POIの読み込み
    poi_df = pd.read_csv(POI_CLUSTERS_CSV)
    
    # 景観データの読み込み (単体GNN出力) とクラスタリング
    ls_nodes = pd.read_csv(LS_NODES_CSV)
    ls_embs = np.load(LS_EMB_NPY)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    labels = kmeans.fit_predict(ls_embs)
    ls_nodes['cluster'] = labels
    ls_valid_df = ls_nodes.copy()

    print(f"POIデータ: {len(poi_df)}件")
    print(f"景観データ: {len(ls_valid_df)}件")

    # 1. 景観クラスタのマップ
    m_ls = folium.Map(location=[41.7687, 140.7288], zoom_start=13)
    print("景観クラスタのマップを生成中...")
    ls_group = folium.FeatureGroup(name="Landscape Clusters (GNN)")
    for _, row in ls_valid_df.iterrows():
        color = get_hex_color(row['cluster'], 'tab20')
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"LS Cluster: {row['cluster']}"
        ).add_to(ls_group)
    ls_group.add_to(m_ls)
    folium.LayerControl().add_to(m_ls)
    
    ls_output_path = OUTPUT_DIR / 'visualize_landscape_clusters.html'
    m_ls.save(str(ls_output_path))
    print(f"景観クラスタのマップを生成しました: {ls_output_path}")

    # 2. POIクラスタのマップ
    m_poi = folium.Map(location=[41.7687, 140.7288], zoom_start=13)
    print("POIクラスタのマップを生成中...")
    poi_group = folium.FeatureGroup(name="POI Clusters (Descriptions)")
    for _, row in poi_df.iterrows():
        color = get_color(row['cluster'])
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"POI: {row['name']}<br>Cluster: {row['cluster']}",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(poi_group)
    poi_group.add_to(m_poi)
    folium.LayerControl().add_to(m_poi)

    poi_output_path = OUTPUT_DIR / 'visualize_poi_clusters.html'
    m_poi.save(str(poi_output_path))
    print(f"POIクラスタのマップを生成しました: {poi_output_path}")

if __name__ == "__main__":
    main()
