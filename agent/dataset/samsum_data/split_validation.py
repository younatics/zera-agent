import os
import pandas as pd

# Paths
src = 'validation.csv'
out_dir = 'validation_chunks'

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Read the CSV
full_path = os.path.join(os.path.dirname(__file__), src)
df = pd.read_csv(full_path)

# Split into 10 chunks of 30 rows each (excluding header)
chunk_size = 30
num_chunks = 10
for i in range(num_chunks):
    chunk = df.iloc[i*chunk_size:(i+1)*chunk_size]
    out_path = os.path.join(os.path.dirname(__file__), out_dir, f'validation_{i+1}.csv')
    chunk.to_csv(out_path, index=False)

print(f"Split into {num_chunks} files of {chunk_size} rows each in '{out_dir}' folder.") 