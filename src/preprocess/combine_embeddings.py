import json
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

# --- Configuration ---
# Alpha: Balances the influence of text vs. images.
# 0.0 = 100% image, 1.0 = 100% text.
ALPHA = 0.8

# Sigma (in meters): Controls how quickly the geographic weight falls off.
# A smaller sigma means only very close images have a high weight.
# A larger sigma allows more distant images to have an influence.
SIGMA = 200.0

# Similarity Threshold: Images with a cosine similarity below this value will be ignored.
# This helps filter out noisy or irrelevant images.
SIMILARITY_THRESHOLD = 0.23

# --- Helper Functions ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two lat/lon points in meters."""
    R = 6371000  # Radius of Earth in meters
    
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

def gaussian_weight(distance, sigma):
    """Calculate weight based on a Gaussian function."""
    return np.exp(-distance**2 / (2 * sigma**2))

# --- Main Logic ---

def combine_embeddings(project_root, alpha, sigma, similarity_threshold):
    """
    Combines text and image embeddings using semantic similarity and geographic distance.
    """
    print("Starting the embedding combination process...")
    print(f"Parameters: alpha={alpha}, sigma={sigma}m, similarity_threshold={similarity_threshold}")

    # --- 1. Define Paths ---
    poi_data_path = os.path.join(project_root, 'data', 'processed', 'poi', 'filtered_facilities.json')
    facility_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'facility_embeddings.npy')
    
    image_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'image_embeddings.npy')
    image_filenames_path = os.path.join(project_root, 'data', 'processed', 'images', 'image_filenames.json')
    # Corrected path for image GPS data as per user feedback
    image_gps_path = os.path.join(project_root, 'data', 'processed', 'images', 'image_gps_data.json')
    
    output_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', f'combined_facility_embeddings_023_08.npy')

    # --- 2. Load All Data ---
    print("Loading data files...")
    try:
        with open(poi_data_path, 'r', encoding='utf-8') as f:
            poi_data = json.load(f)
        with open(image_filenames_path, 'r', encoding='utf-8') as f:
            image_filenames = json.load(f)
        with open(image_gps_path, 'r', encoding='utf-8') as f:
            image_gps_data = json.load(f)
            
        facility_embeddings = np.load(facility_emb_path)
        image_embeddings = np.load(image_emb_path)
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        print("Please ensure all previous steps have been completed successfully.")
        return

    # --- 3. Prepare Data Structures for Fast Lookup ---
    image_gps_list = [image_gps_data.get(fname) for fname in image_filenames]

    new_facility_embeddings = []

    print(f"Processing {len(poi_data)} facilities...")
    # --- 4. Main Processing Loop ---
    for i, facility in enumerate(tqdm.tqdm(poi_data, desc="Combining embeddings")):
        text_embedding = facility_embeddings[i]
        poi_location = facility.get('google_places_data', {}).get('find_place_geometry', {}).get('location')

        if not poi_location:
            new_facility_embeddings.append(text_embedding)
            continue

        # --- 4a. Calculate Semantic Similarity Weights (w_sim) ---
        text_embedding_reshaped = text_embedding.reshape(1, -1)
        w_sim = cosine_similarity(text_embedding_reshaped, image_embeddings).flatten()

        # --- 4b. Calculate Geographic Weights (w_geo) ---
        w_geo = np.zeros_like(w_sim)
        for j, gps in enumerate(image_gps_list):
            if gps and 'lat' in gps and 'lon' in gps:
                distance = haversine_distance(poi_location['lat'], poi_location['lng'], gps['lat'], gps['lon'])
                w_geo[j] = gaussian_weight(distance, sigma)

        # --- 4c. Combine Weights and Normalize ---
        # Apply the similarity threshold to filter out irrelevant images
        similarity_mask = w_sim > similarity_threshold
        w_final = w_sim * w_geo
        w_final[~similarity_mask] = 0  # Set weight to 0 for images below the threshold
        
        weight_sum = np.sum(w_final)
        if weight_sum > 0:
            w_final_normalized = w_final / weight_sum
        else:
            new_facility_embeddings.append(text_embedding)
            continue

        # --- 4d. Synthesize the Image Vector ---
        w_final_reshaped = w_final_normalized.reshape(-1, 1)
        combined_image_embedding = np.sum(image_embeddings * w_final_reshaped, axis=0)

        # --- 4e. Final Combination ---
        new_embedding = alpha * text_embedding + (1 - alpha) * combined_image_embedding
        new_facility_embeddings.append(new_embedding)

    # --- 5. Save the Result ---
    final_embeddings_array = np.array(new_facility_embeddings)
    np.save(output_path, final_embeddings_array)

    print("\n--- Combination Complete ---")
    print(f"Successfully generated {final_embeddings_array.shape[0]} new embeddings.")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    try:
        import sklearn
        import tqdm
    except ImportError:
        print("Error: Required libraries 'scikit-learn' or 'tqdm' not found.")
        print("Please install them using: pip install scikit-learn tqdm")
    else:
        combine_embeddings(project_root_dir, ALPHA, SIGMA, SIMILARITY_THRESHOLD)