# This script combines the functionality of point generation and bounding box filtering.
# It performs the following steps:
# 1. Reads an OpenStreetMap (OSM) XML file.
# 2. Generates points at a fixed interval (e.g., 50m) along each road ('way').
# 3. Filters these roads, keeping only those that have at least one point within a
#    specified bounding box (BBOX).
# 4. Saves the filtered, road-centric data to a new JSON file.

import os
import json
import xml.etree.ElementTree as ET
from geopy.distance import great_circle, Point
import sys

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONFIGURATION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
INPUT_OSM_FILE = './data/raw/osm_hakodate/Hakodate.osm.xml'
OUTPUT_DIR = './data/processed/road_points'
OUTPUT_FILTERED_FILE = os.path.join(OUTPUT_DIR, 'road_points_per_way_50m_filtered.json')

INTERVAL_METERS = 50
COORDINATE_PRECISION = 6

# Bounding Box [lat_min, lon_min, lat_max, lon_max]
# From filter_points_by_bbox.py
BBOX = [
    41.73987856651839,  # lat_min
    140.6939413999981,  # lon_min
    41.78479021564596,  # lat_max
    140.73818711897565   # lon_max
]

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# HELPER FUNCTIONS (from generate_road_and_interval_points_only.py)
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def build_node_cache(osm_file):
    """Parses the OSM XML file once to build a dictionary of node IDs to their coordinates."""
    print(f"Building node cache from {osm_file}...")
    nodes = {}
    context = ET.iterparse(osm_file, events=("start",))
    for _, elem in context:
        if elem.tag == 'node':
            try:
                nodes[elem.attrib['id']] = (float(elem.attrib['lat']), float(elem.attrib['lon']))
            except (KeyError, ValueError):
                pass
        elem.clear()
    print(f"Node cache built with {len(nodes)} nodes.")
    return nodes

def get_way_path_and_length(way_elem, node_cache):
    """Gets the coordinate path for a way and calculates its total length."""
    node_ids = [nd.attrib['ref'] for nd in way_elem.findall('nd')]
    path = [node_cache[nid] for nid in node_ids if nid in node_cache]
    length = sum(great_circle(path[i], path[i+1]).meters for i in range(len(path) - 1))
    return path, length

def get_points_on_path(path, interval):
    """Generates candidate points along a coordinate path at a given interval."""
    if not path: return []
    if len(path) < 2: return [path[0]]

    points = [path[0]]
    dist_to_next_marker = interval
    
    for i in range(len(path) - 1):
        start_node = Point(path[i])
        end_node = Point(path[i+1])
        segment_dist = great_circle(start_node, end_node).meters
        
        if segment_dist > 0:
            while dist_to_next_marker <= segment_dist:
                fraction = dist_to_next_marker / segment_dist
                lat = start_node.latitude + fraction * (end_node.latitude - start_node.latitude)
                lon = start_node.longitude + fraction * (end_node.longitude - start_node.longitude)
                points.append((lat, lon))
                dist_to_next_marker += interval
            dist_to_next_marker -= segment_dist
    
    if points[-1] != path[-1]:
        points.append(path[-1])
            
    return points

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# MAIN EXECUTION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def main():
    print("--- Starting Combined Point Generation and Filtering Script ---")
    
    # --- Folder Preparation ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # ======================================================================
    # Phase 1: Generate points for all roads (in-memory)
    # ======================================================================
    node_cache = build_node_cache(INPUT_OSM_FILE)
    
    road_points_dict = {}
    processed_ways = 0
    
    print(f"\n--- Phase 1: Processing roads from {INPUT_OSM_FILE} ---")
    context = ET.iterparse(INPUT_OSM_FILE, events=("start",))
    
    for _, elem in context:
        if elem.tag == 'way' and any(tag.attrib.get('k') == 'highway' for tag in elem.findall('tag')):
            processed_ways += 1
            way_id = elem.attrib['id']
            
            path, length = get_way_path_and_length(elem, node_cache)
            if not path: continue
            
            generated_points = get_points_on_path(path, INTERVAL_METERS)
            
            rounded_points = [(round(lat, COORDINATE_PRECISION), round(lon, COORDINATE_PRECISION)) for lat, lon in generated_points]
            
            if rounded_points:
                road_points_dict[way_id] = rounded_points
            
            if (processed_ways % 2000) == 0:
                print(f"  Processed {processed_ways} ways...")
        elem.clear()

    print(f"\n--- Point Generation Complete ---")
    print(f"Total roads with generated points: {len(road_points_dict)}")

    # ======================================================================
    # Phase 2: Filter the generated roads by BBOX
    # =================================================0=====================
    print("\n--- Phase 2: Filtering generated roads by Bounding Box ---")
    print(f"BBOX (lat_min, lon_min, lat_max, lon_max) = {BBOX}")
    lat_min, lon_min, lat_max, lon_max = BBOX
    
    filtered_road_points_dict = {}
    for road_id, points in road_points_dict.items():
        # Keep the road if ANY of its points are within the BBOX
        if any(lat_min <= lat <= lat_max and lon_min <= lon <= lon_max for lat, lon in points):
            filtered_road_points_dict[road_id] = points
    
    print(f"\n--- Filtering Complete ---")
    print(f"Original number of roads: {len(road_points_dict)}")
    print(f"Filtered number of roads within BBOX: {len(filtered_road_points_dict)}")
    
    # ======================================================================
    # Phase 3: Save the filtered data
    # ======================================================================
    print(f"\n--- Phase 3: Saving filtered data ---")
    print(f"Saving filtered points dictionary to {OUTPUT_FILTERED_FILE}...")
    try:
        with open(OUTPUT_FILTERED_FILE, 'w', encoding='utf-8') as f:
            json.dump(filtered_road_points_dict, f, indent=2, ensure_ascii=False)
        print("--- Successfully saved filtered points. ---")
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
