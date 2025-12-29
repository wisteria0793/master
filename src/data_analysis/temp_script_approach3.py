
import pandas as pd
import re

# Load the CSV file
file_path = 'data/processed/segmentation_results_50m/segmentation_ratios.csv'
df = pd.read_csv(file_path)

# Extract point_id and direction (heading) from filename
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

# Drop original filename and heading columns, keep point_id and direction
df_final = df.drop(columns=['filename', 'heading'])

# Fill NaN values with 0
df_final.fillna(0, inplace=True)

# Reorder columns to have point_id and direction at the beginning
# Get all columns except point_id and direction, then prepend them
other_cols = [col for col in df_final.columns if col not in ['point_id', 'direction']]
df_final = df_final[['point_id', 'direction'] + other_cols]


# Save the result
output_path = 'data/processed/segmentation_results_50m/separate_view_vectors.csv'
df_final.to_csv(output_path, index=False)

print(f"Successfully created separate view vectors at: {output_path}")
print("\nFirst 5 rows of the new file:")
print(df_final.head())
