import json
import os

def filter_points_by_bbox():
    """
    Filters a list of coordinate points to include only those within a
    specified bounding box.
    """
    # --- Configuration ---
    INPUT_FILE = './data/processed/road_points/road_points_50m_from_net.json'
    OUTPUT_FILE = './data/processed/road_points/road_points_50m_filtered.json'
    
    # Bounding Box [lat_min, lon_min, lat_max, lon_max]
    # Defined from user input: (41.7847..., 140.6939...) and (41.7398..., 140.7381...)
    BBOX = [
        41.73987856651839,  # lat_min
        140.6939413999981,  # lon_min
        41.78479021564596,  # lat_max
        140.73818711897565   # lon_max
    ]
    
    lat_min, lon_min, lat_max, lon_max = BBOX
    
    print(f"Loading points from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_points = json.load(f)
        print(f"Loaded {len(all_points)} total points.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
        
    print(f"Filtering points within bounding box: {BBOX}...")
    
    filtered_points = [
        point for point in all_points 
        if lat_min <= point[0] <= lat_max and lon_min <= point[1] <= lon_max
    ]
    
    print(f"Found {len(filtered_points)} points within the specified area.")

    print(f"Saving filtered points to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(filtered_points, f, ensure_ascii=False, indent=2)
        print("--- Successfully filtered and saved points. ---")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    filter_points_by_bbox()
