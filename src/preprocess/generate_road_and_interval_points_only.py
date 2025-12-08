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
OUTPUT_POINTS_FILE = os.path.join(OUTPUT_DIR, 'road_and_interval_points_50m.json')

INTERVAL_METERS = 50  # The target interval for sampling along roads
COORDINATE_PRECISION = 6 # Number of decimal places for coordinates

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# HELPER FUNCTIONS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def build_node_cache(osm_file):
    """
    Parses the OSM XML file once to build a dictionary of node IDs to their coordinates.
    Uses iterparse for memory efficiency.
    """
    print(f"Building node cache from {osm_file}...")
    nodes = {}
    context = ET.iterparse(osm_file, events=("start",))
    for _, elem in context:
        if elem.tag == 'node':
            try:
                node_id = elem.attrib['id']
                lat = float(elem.attrib['lat'])
                lon = float(elem.attrib['lon'])
                nodes[node_id] = (lat, lon)
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping malformed node. Error: {e}", file=sys.stderr)
        # Clear the element to free memory
        elem.clear()
    print(f"Node cache built with {len(nodes)} nodes.")
    return nodes

def get_way_path_and_length(way_elem, node_cache):
    """Gets the coordinate path for a way and calculates its total length."""
    node_ids = [nd.attrib['ref'] for nd in way_elem.findall('nd')]
    path = [node_cache[nid] for nid in node_ids if nid in node_cache]
    
    length = 0.0
    if len(path) > 1:
        for i in range(len(path) - 1):
            length += great_circle(path[i], path[i+1]).meters
        
    return path, length

def get_points_on_path(path, interval):
    """Generates candidate points along a coordinate path at a given interval."""
    if not path:
        return []
    if len(path) < 2: # Road with only one node
        return [path[0]]

    points = [path[0]] # Always include the start point
    dist_to_next_marker = interval
    
    for i in range(len(path) - 1):
        start_node = Point(path[i])
        end_node = Point(path[i+1])
        segment_dist = great_circle(start_node, end_node).meters
        
        # Check if the segment itself is shorter than the interval and we haven't placed a point
        # This handles cases where a segment is very short, ensuring points are still added.
        if segment_dist > 0:
            while dist_to_next_marker <= segment_dist:
                fraction = dist_to_next_marker / segment_dist
                lat = start_node.latitude + fraction * (end_node.latitude - start_node.latitude)
                lon = start_node.longitude + fraction * (end_node.longitude - start_node.longitude)
                points.append((lat, lon))
                
                dist_to_next_marker += interval # Move to the next interval marker
            
            # Carry over the remaining distance for the next segment
            dist_to_next_marker = dist_to_next_marker - segment_dist
            
        else: # Handle zero-length segments if any
            dist_to_next_marker -= segment_dist
            
    # Always include the end point if it's not already covered
    if path[-1] not in points: # Check for exact match, relies on floating point precision
        # A better check would be proximity, but for OSM nodes, exact match is often ok
        # For this script's purpose, the last point from path should be considered
        points.append(path[-1])
            
    return points

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# MAIN EXECUTION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def main():
    print("--- Starting Road and Interval Point Generator (Per-Road Structure) ---")
    
    # --- Update Output Filename ---
    OUTPUT_POINTS_FILE = os.path.join(OUTPUT_DIR, 'road_points_per_way_50m.json')

    # --- Folder Preparation ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    node_cache = build_node_cache(INPUT_OSM_FILE)
    
    road_points_dict = {} # Use a dictionary to store points per way_id
    processed_ways = 0
    total_points_generated = 0

    print(f"\n--- Processing roads from {INPUT_OSM_FILE} to generate points per road ---")
    context = ET.iterparse(INPUT_OSM_FILE, events=("start",))
    
    for _, elem in context:
        if elem.tag == 'way':
            # Check if it's a road (has 'highway' tag)
            is_road = any(tag.attrib.get('k') == 'highway' for tag in elem.findall('tag'))
            if not is_road:
                elem.clear() # Free memory
                continue
            
            processed_ways += 1
            way_id = elem.attrib['id']
            
            path, length = get_way_path_and_length(elem, node_cache)
            if not path:
                print(f"Warning: Way #{way_id} has no valid path. Skipping.", file=sys.stderr)
                elem.clear()
                continue
            
            # Generate primary candidate points for the way
            generated_points_for_way = get_points_on_path(path, INTERVAL_METERS)
            
            # Round coordinates and store them under the way_id
            rounded_points = [
                (round(lat, COORDINATE_PRECISION), round(lon, COORDINATE_PRECISION))
                for lat, lon in generated_points_for_way
            ]
            
            if rounded_points:
                road_points_dict[way_id] = rounded_points
                total_points_generated += len(rounded_points)
            
            if (processed_ways % 1000) == 0:
                print(f"  Processed {processed_ways} ways...")

            elem.clear() # Free memory
    
    print(f"\n--- Point Generation Complete ---")
    print(f"Total ways processed: {processed_ways}")
    print(f"Total points generated across all ways: {total_points_generated}")
    print(f"Total roads with points: {len(road_points_dict)}")
    
    print(f"Saving points dictionary to {OUTPUT_POINTS_FILE}...")
    try:
        with open(OUTPUT_POINTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(road_points_dict, f, indent=2, ensure_ascii=False)
        print("--- Successfully saved all generated points. ---")
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
