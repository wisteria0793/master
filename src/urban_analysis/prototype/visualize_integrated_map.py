# -*- coding: utf-8 -*-
"""
統合地図可視化スクリプト
景観クラスターとPOIクラスターの組み合わせを可視化し、
典型的なエリアと意外性のある（乖離した）エリアを色分けする。
"""

import pandas as pd
import folium
from folium.plugins import MarkerCluster
import numpy as np
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
INTEGRATED_DATA = PROJECT_ROOT / 'data' / 'processed' / 'integrated_poi_landscape_clusters.csv'
OUTPUT_MAP = PROJECT_ROOT / 'docs' / 'results' / 'integrated_urban_map.html'

def get_color(is_dissonance):
    # 典型的な場所は青、意外性のある場所は赤
    return 'red' if is_dissonance else 'blue'

def main():
    print("統合データを読み込み中...")
    df = pd.read_csv(INTEGRATED_DATA)
    
    # 景観データが紐付いていないPOIを除外
    df_valid = df[df['dominant_landscape_cluster'] != -1].copy()
    
    # 1. 典型的な組み合わせ（共生行列）の算出
    # 各 text_cluster に対して、最も頻繁に現れる landscape_cluster を「典型的」と定義
    co_occurrence = df_valid.groupby(['cluster', 'dominant_landscape_cluster']).size().unstack(fill_value=0)
    typical_landscape_per_text = co_occurrence.idxmax(axis=1).to_dict()
    
    # 2. 乖離（Dissonance）の判定
    df_valid['is_dissonance'] = df_valid.apply(
        lambda row: row['dominant_landscape_cluster'] != typical_landscape_per_text.get(row['cluster'], -1),
        axis=1
    )
    
    # 3. マップの生成
    hakodate_center = [41.768, 140.729]
    m = folium.Map(location=hakodate_center, zoom_start=14, tiles='cartodbpositron')
    
    # マーカークラスターの追加
    marker_cluster = MarkerCluster().add_to(m)
    
    for i, row in df_valid.iterrows():
        color = get_color(row['is_dissonance'])
        label = "意外なスポット" if row['is_dissonance'] else "標準的なスポット"
        
        popup_html = f"""
        <div style="width: 250px;">
            <h4>{row['name']}</h4>
            <b>タイプ:</b> {label}<br>
            <b>機能クラスター ID:</b> {int(row['cluster'])}<br>
            <b>周辺景観クラスター ID:</b> {int(row['dominant_landscape_cluster'])}<br>
            <hr>
            <p style="font-size: 12px;">{row.get('categories', 'カテゴリ情報なし')}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2
        ).add_to(m)
        
    # 保存
    os.makedirs(OUTPUT_MAP.parent, exist_ok=True)
    m.save(str(OUTPUT_MAP))
    print(f"統合可視化マップを保存しました: {OUTPUT_MAP}")
    print(f"統計: 典型地点 {len(df_valid[~df_valid['is_dissonance']])} / 意外地点 {len(df_valid[df_valid['is_dissonance']])}")

if __name__ == "__main__":
    import os
    main()
