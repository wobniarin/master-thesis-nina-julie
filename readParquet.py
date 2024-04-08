import pyarrow.parquet as pq

# Path to your large Parquet file
file_path = 'c:\\Users\\Julie Abildgaard\\OneDrive\\ITU\\Kandidat\\Code\\master-thesis-nina-julie\\data\\features\\US-CAL-CISO.parquet'

# Open the Parquet file
dataset = pq.ParquetDataset(file_path)

# Read in chunks
for chunk in dataset.read():
    # Process each chunk
    print(chunk)