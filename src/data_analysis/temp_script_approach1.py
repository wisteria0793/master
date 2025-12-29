import pandas as pd
import re

# Load the CSV file
file_path = 'data/processed/segmentation_results_50m/segmentation_ratios.csv'
df = pd.read_csv(file_path)

# Extract point_id and direction from filename
def extract_info(filename):
    match = re.search(r'pano_([a-zA-Z0-9_-]+)_h(\d+)\.jpg', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

df[['point_id', 'heading']] = df['filename'].apply(lambda x: pd.Series(extract_info(x)))

# Drop rows where info extraction failed
df.dropna(subset=['point_id', 'heading'], inplace=True)

# Map heading to direction names for clarity
direction_map = {
    '0': 'front',
    '90': 'right',
    '180': 'back',
    '270': 'left'
}
df['direction'] = df['heading'].map(direction_map)

# Drop original filename and heading columns
df = df.drop(columns=['filename', 'heading'])

# Set index for pivot
df.set_index(['point_id', 'direction'], inplace=True)

# Pivot the table to concatenate vectors
concatenated_df = df.unstack(level='direction')

# --- ADDED STEP: Fill NaN values with 0 ---
concatenated_df.fillna(0, inplace=True)

# Flatten the multi-level column names
concatenated_df.columns = [f'{col[0]}_{col[1]}' for col in concatenated_df.columns]

# Reset index to make point_id a column again
concatenated_df.reset_index(inplace=True)

# Save the result
output_path = 'data/processed/segmentation_results_50m/concatenated_vectors.csv'
concatenated_df.to_csv(output_path, index=False)

print(f"Successfully created concatenated vectors at: {output_path}")
print(f"NaN values have been filled with 0.")
print(f"Verification: Total NaN values in the final file = {concatenated_df.isnull().sum().sum()}")
print("\nFirst 5 rows of the new file:")
print(concatenated_df.head())