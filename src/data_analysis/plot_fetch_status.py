import sys
import json
import pandas as pd

try:
    import folium
except ImportError:
    sys.exit("folium library not found. Please install with 'pip install folium'")

def plot_fetch_status(log_file, roads_file, output_html):
    """
    画像収集のログを読み込み、結果をインタラクティブな地図として可視化する
    """
    print(f"Reading log file: {log_file}")
    try:
        df = pd.read_json(log_file)
    except Exception as e:
        sys.exit(f"Error reading log file: {e}")

    if df.empty:
        sys.exit("Log file is empty. Nothing to plot.")

    print(f"Reading roads file: {roads_file}")
    try:
        with open(roads_file, 'r', encoding='utf-8') as f:
            roads_data = list(json.load(f).values())
    except Exception as e:
        print(f"Warning: Could not read roads file: {e}")
        roads_data = []

    # 地図の中心を計算
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    # 地図を初期化
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

    # 1. 背景の道路網をプロット
    if roads_data:
        print("Plotting road network...")
        road_group = folium.FeatureGroup(name="Road Network")
        for road in roads_data:
            if len(road) > 1:
                folium.PolyLine(
                    locations=road,
                    color='gray',
                    weight=1.5,
                    opacity=0.5
                ).add_to(road_group)
        road_group.add_to(m)

    # 2. 収集結果の「added」ポイントのみをプロット
    print("Plotting only 'added' fetch status points...")

    # DataFrameをフィルタリングして'added'ステータスのポイントのみを抽出
    df_added = df[df['status'] == 'added'].copy() # SettingWithCopyWarningを避けるために.copy()を使用

    if df_added.empty:
        print("No 'added' points found to plot.")
    else:
        # 'added'ポイント専用のフィーチャーグループを作成
        added_group = folium.FeatureGroup(name="Added Points (Successful)")

        for _, row in df_added.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.7,
                popup=f"Status: {row['status']}<br>Type: {row['type']}<br>Lat: {row['lat']:.5f}<br>Lon: {row['lon']:.5f}"
            ).add_to(added_group)

        added_group.add_to(m)

    # レイヤーコントロールを追加
    folium.LayerControl().add_to(m)

    # 地図をHTMLファイルとして保存
    m.save(output_html)
    print(f"\nSuccessfully created visualization map: {output_html}")
    print(f"Open this file in your web browser to see the interactive map.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_fetch_status.py <path_to_fetch_log.json>")
        print("Example: python src/data_analysis/plot_fetch_status.py data/raw/street_view_images_test/fetch_log.json")
        sys.exit(1)
        
    log_file = sys.argv[1]
    roads_file = 'data/processed/road_points/road_points_per_way_50m.json'
    output_html = 'fetch_status_map.html'
    
    plot_fetch_status(log_file, roads_file, output_html)

if __name__ == "__main__":
    main()
