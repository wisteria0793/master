import json
import os
import folium
import pandas as pd

def create_visualization_map():
    """
    Creates an interactive map to visualize the road network and the generated points.
    """
    # --- Configuration ---
    ROAD_DATA_FILE = './data/raw/osm_hakodate/hakodate_osm_raw.json'
    POINTS_FILE = './data/processed/road_points/road_points_50m_filtered.json'
    OUTPUT_FILE = './docs/results/road_points_visualization_50m_filtered.html'
    
    # BBox for filtering roads to display
    BBOX = [41.73987856651839, 140.6939413999981, 41.78479021564596, 140.73818711897565]
    lat_min, lon_min, lat_max, lon_max = BBOX
    
    # Plot all filtered points and relevant roads
    PLOT_EVERY_NTH_ROAD = 1
    PLOT_EVERY_NTH_POINT = 1

    # --- Load Data ---
    print(f"Loading generated points from {POINTS_FILE}...")
    try:
        with open(POINTS_FILE, 'r', encoding='utf-8') as f:
            points_to_plot = json.load(f)
    except FileNotFoundError:
        print(f"Error: Points file not found at {POINTS_FILE}")
        return
        
    print(f"Loading OSM road data from {ROAD_DATA_FILE}...")
    try:
        with open(ROAD_DATA_FILE, 'r', encoding='utf-8') as f:
            osm_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: OSM data file not found at {ROAD_DATA_FILE}")
        return

    # --- Create Base Map ---
    if not points_to_plot:
        print("No points to plot. Centering on a default Hakodate location.")
        map_center = [41.7687, 140.7288] # Default to Hakodate station
    else:
        df = pd.DataFrame(points_to_plot, columns=['lat', 'lon'])
        map_center = [df['lat'].mean(), df['lon'].mean()]
        
    print(f"Creating map centered at {map_center}...")
    m = folium.Map(location=map_center, zoom_start=14) # Zoom in closer

    # --- Plot Roads ---
    print("Filtering and plotting road network within the bounding box...")
    nodes_lookup = {node['id']: (node['lat'], node['lon']) for node in osm_data.get('nodes', [])}
    roads = [way for way in osm_data.get('ways', []) if 'highway' in way.get('tags', {})]
    
    plotted_road_count = 0
    for i, road in enumerate(roads):
        if i % PLOT_EVERY_NTH_ROAD == 0:
            way_points = [nodes_lookup[node_id] for node_id in road.get('nodes', []) if node_id in nodes_lookup]
            
            # Check if any part of the road is within the bbox to decide whether to plot it
            is_road_in_bbox = False
            for lat, lon in way_points:
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    is_road_in_bbox = True
                    break
            
            if is_road_in_bbox and len(way_points) >= 2:
                folium.PolyLine(
                    locations=way_points,
                    color='blue',
                    weight=1.5,
                    opacity=0.8
                ).add_to(m)
                plotted_road_count += 1
    print(f"Plotted {plotted_road_count} road segments.")

    # --- Plot Generated Points ---
    print(f"Plotting generated points...")
    plotted_point_count = 0
    for i, point in enumerate(points_to_plot):
        if i % PLOT_EVERY_NTH_POINT == 0:
            folium.CircleMarker(
                location=point,
                radius=2,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.8
            ).add_to(m)
            plotted_point_count += 1
    print(f"Plotted {plotted_point_count} generated points.")

    # --- Save Map ---
    print(f"Saving map to {OUTPUT_FILE}...")
    try:
        m.save(OUTPUT_FILE)
        print("--- Successfully created visualization map. ---")
        print(f"You can now open '{OUTPUT_FILE}' in a web browser.")
    except Exception as e:
        print(f"Error saving map: {e}")

if __name__ == "__main__":
    create_visualization_map()
