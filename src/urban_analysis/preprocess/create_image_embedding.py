import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob

def create_image_embeddings(image_dir, output_dir, model_name='openai/clip-vit-base-patch32', batch_size=32):
    """
    Generates embeddings for all images in a directory using a CLIP model
    and saves them to a .npy file. Also saves the corresponding filenames.
    """
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the CLIP model and processor
    try:
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and the 'transformers' library is installed.")
        return

    # Find all image files in the directory
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
                  sorted(glob.glob(os.path.join(image_dir, '*.jpeg'))) + \
                  sorted(glob.glob(os.path.join(image_dir, '*.png')))

    if not image_paths:
        print(f"No images found in directory: {image_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    all_embeddings = []
    all_filenames = []

    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing image batches"):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(path) for path in batch_paths]
        
        try:
            # Preprocess the images
            inputs = processor(text=None, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get image features (embeddings)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            all_embeddings.append(image_features.cpu().numpy())
            all_filenames.extend([os.path.basename(path) for path in batch_paths])

        except Exception as e:
            print(f"Error processing batch starting with {batch_paths[0]}: {e}")
            # Optionally, skip the batch and continue
            continue

    # Concatenate all embeddings into a single numpy array
    if not all_embeddings:
        print("No embeddings were generated.")
        return
        
    embeddings_array = np.vstack(all_embeddings)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output paths
    embeddings_output_path = os.path.join(output_dir, 'image_embeddings.npy')
    filenames_output_path = os.path.join(output_dir, 'image_filenames.json')

    # Save the embeddings and filenames
    np.save(embeddings_output_path, embeddings_array)
    with open(filenames_output_path, 'w') as f:
        json.dump(all_filenames, f, indent=4)

    print(f"Successfully saved {len(all_filenames)} image embeddings to: {embeddings_output_path}")
    print(f"Filenames saved to: {filenames_output_path}")


if __name__ == '__main__':
    # Define absolute paths for data
    # IMPORTANT: This script assumes it is run from the project root directory.
    # For robustness, consider using a more dynamic path resolution method.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    image_directory = os.path.join(project_root, 'data', 'raw', 'hakodate_all_photos_bbox')
    output_directory = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip')

    create_image_embeddings(image_directory, output_directory)
