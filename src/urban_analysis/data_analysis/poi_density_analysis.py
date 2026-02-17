
import json
import os
import sys
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from shapely.geometry import LineString
from tqdm import tqdm
from pathlib import Path

# プロジェクトルートをパスに追加して config をインポート可能にする
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.urban_analysis.config import OSM_XML_PATH, RAW_DATA_DIR, PROJECT_ROOT
from src.urban_analysis.data_analysis.network_kde import network_kernel_density

# --- 設定 ---
JSON_PATH = RAW_DATA_DIR / 'output_with_google_places_jp.json'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results' / 'poi'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BANDWIDTH = 300  # メートル
THRESHOLD_DENSITY = 0.01 # 表示閾値

def load_poi_data(json_path):
    """JSONファイルからPOIデータを読み込み、DataFrameに変換する"""
    print(f"POIデータを読み込み中: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    poi_list = []
    for item in data:
        name = item.get('name')
        google_places = item.get('google_places_data', {})
        
        # 緯度経度情報の取得
        # find_place_geometryがある場合とない場合があるかもしれないので注意
        geometry = google_places.get('find_place_geometry', {})
        location = geometry.get('location', {})
        lat = location.get('lat')
        lng = location.get('lng')
        
        if lat is not None and lng is not None:
            poi_list.append({
                'name': name,
                'lat': lat,
                'lng': lng,
                'address': item.get('address'),
                'categories': item.get('categories', [])
            })
    
    df = pd.DataFrame(poi_list)
    print(f"有効なPOIデータ数: {len(df)}")
    return df

def main():
    # 1. POIデータの読み込み
    poi_df = load_poi_data(JSON_PATH)
    if poi_df.empty:
        print("有効なPOIデータが見つかりませんでした。終了します。")
        return

    # 2. 道路ネットワークの読み込み
    print(f"道路ネットワークを読み込み中: {OSM_XML_PATH}")
    if not OSM_XML_PATH.exists():
        print(f"エラー: OSMファイルが見つかりません: {OSM_XML_PATH}")
        return
        
    G = ox.graph_from_xml(OSM_XML_PATH, simplify=True)
    
    # 3. POIをネットワーク上の最近傍ノードにマッピング
    print("POIをネットワークノードにマッピング中...")
    # ox.nearest_nodes は (X, Y) = (lng, lat) の順序
    nearest_nodes = ox.nearest_nodes(G, poi_df['lng'].values, poi_df['lat'].values)
    poi_df['osm_node'] = nearest_nodes
    
    # 4. Network KDEの実行
    print(f"Network KDEを実行中 (Bandwidth={BANDWIDTH}m)...")
    edge_densities = network_kernel_density(G, poi_df, BANDWIDTH, node_column='osm_node')
    
    density_series = pd.Series(edge_densities)
    
    # 5. 可視化
    print("結果を可視化中...")
    
    # ベースマップの中心設定
    center_lat = poi_df['lat'].mean()
    center_lng = poi_df['lng'].mean()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles='CartoDB positron')

    # 密度の正規化とカラーマッピング
    if not density_series.empty and density_series.max() > 0:
        # 閾値以上の密度を持つエッジのみ抽出
        valid_densities = density_series[density_series >= THRESHOLD_DENSITY]
        
        if not valid_densities.empty:
            # カラーマップの作成 (Plasmaを使用)
            # LogNormを使用して、密度の低い部分と高い部分の差を視覚化しやすくする
            vmin = valid_densities.min()
            vmax = valid_densities.max()
            norm = LogNorm(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('plasma')
            
            print(f"密度の範囲: {vmin:.6f} - {vmax:.6f}")

            # エッジの描画
            for u, v, key, data in G.edges(keys=True, data=True):
                # MultiGraph対応のキー生成
                edge_key = (u, v, key)
                
                # 密度を取得 (MultiGraphでない場合は (u, v, 0) で取得される可能性があるため注意)
                # network_kdeの実装に合わせて取得
                density = density_series.get(edge_key, 0.0)
                
                if density >= THRESHOLD_DENSITY:
                    # 色の決定
                    color_rgba = cmap(norm(density))
                    color_hex = '#%02x%02x%02x' % (int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))
                    
                    # ジオメトリの取得
                    if 'geometry' in data:
                        edge_geom = data['geometry']
                        points = [(lat, lng) for lng, lat in edge_geom.coords]
                    else:
                        point_u = (G.nodes[u]['y'], G.nodes[u]['x'])
                        point_v = (G.nodes[v]['y'], G.nodes[v]['x'])
                        points = [point_u, point_v]
                    
                    # 線の太さを密度に応じて変える（オプション）
                    weight = 3 + (norm(density) * 5)
                    
                    folium.PolyLine(
                        locations=points,
                        color=color_hex,
                        weight=weight,
                        opacity=0.8,
                        tooltip=f"Density: {density:.6f}"
                    ).add_to(m)
        else:
            print("閾値を超える密度が見つかりませんでした。")
    else:
        print("密度が計算されませんでした。")

    # # POIのマーカーを追加 (オプション)
    # # クラスターとして表示
    # from folium.plugins import MarkerCluster
    # marker_cluster = MarkerCluster().add_to(m)
    
    # for _, row in poi_df.iterrows():
    #     folium.Marker(
    #         location=[row['lat'], row['lng']],
    #         popup=folium.Popup(f"<b>{row['name']}</b><br>{row['address']}", max_width=300),
    #         icon=folium.Icon(color='blue', icon='info-sign', prefix='fa')
    #     ).add_to(marker_cluster)

    # 保存
    output_file = OUTPUT_DIR / f'poi_density_bw{BANDWIDTH}.html'
    m.save(str(output_file))
    print(f"マップを保存しました: {output_file}")

if __name__ == "__main__":
    main()
