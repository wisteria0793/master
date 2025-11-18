import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

def plot_similarity_distribution(project_root, sample_size=50):
    """
    Calculates cosine similarities between a sample of text embeddings and all
    image embeddings, then plots and saves the distribution as a histogram.
    """
    print("Starting similarity distribution analysis...")

    # --- 1. Define Paths ---
    facility_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'facility_embeddings.npy')
    image_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'image_embeddings.npy')
    output_image_path = os.path.join(project_root, 'images', 'similarity_distribution.png')

    # --- 2. Load Embeddings ---
    print("Loading embedding files...")
    try:
        facility_embeddings = np.load(facility_emb_path)
        image_embeddings = np.load(image_emb_path)
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        print("Please ensure facility and image embeddings have been created.")
        return

    num_facilities = facility_embeddings.shape[0]
    print(f"Found {num_facilities} facilities and {image_embeddings.shape[0]} images.")

    # --- 3. Sample Facilities and Calculate Similarities ---
    if num_facilities > sample_size:
        print(f"Sampling {sample_size} facilities for analysis...")
        facility_indices = random.sample(range(num_facilities), sample_size)
        sampled_facility_embeddings = facility_embeddings[facility_indices]
    else:
        print("Using all facilities for analysis as total is less than sample size.")
        sampled_facility_embeddings = facility_embeddings

    all_similarities = []
    print("Calculating cosine similarities...")
    for text_embedding in tqdm.tqdm(sampled_facility_embeddings, desc="Processing facilities"):
        # Calculate similarities between one text embedding and all image embeddings
        sims = cosine_similarity(text_embedding.reshape(1, -1), image_embeddings)
        all_similarities.extend(sims.flatten())

    # --- 4. Plot the Distribution ---
    print("Plotting the distribution...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.histplot(all_similarities, bins=100, kde=True, ax=ax)
    
    ax.set_title('Distribution of Cosine Similarities between Text and Image Embeddings', fontsize=16)
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    # Calculate and display statistics
    mean_sim = np.mean(all_similarities)
    median_sim = np.median(all_similarities)
    std_sim = np.std(all_similarities)
    
    stats_text = f"Mean: {mean_sim:.3f}\nMedian: {median_sim:.3f}\nStd Dev: {std_sim:.3f}"
    ax.axvline(mean_sim, color='r', linestyle='--', label=f'Mean ({mean_sim:.3f})')
    ax.axvline(median_sim, color='g', linestyle='-', label=f'Median ({median_sim:.3f})')
    
    ax.legend()
    
    # --- 5. Save the Plot ---
    try:
        fig.savefig(output_image_path, dpi=300)
        print(f"\nSuccessfully saved the plot to: {output_image_path}")
    except IOError as e:
        print(f"Error saving plot: {e}")

if __name__ == '__main__':
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Check for required libraries
    try:
        import matplotlib
        import seaborn
        import sklearn
        import tqdm
    except ImportError:
        print("Error: One or more required libraries are not found.")
        print("Please install them using: pip install matplotlib seaborn scikit-learn tqdm")
    else:
        plot_similarity_distribution(project_root_dir)
