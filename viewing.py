import pandas as pd
import pyarrow.parquet as pq

# Specify the path to your Parquet file
path = 'data\\features\\US-TEX-ERCO.parquet'

# Load the entire Parquet file into a pandas DataFrame
table = pq.read_table(path)
df = table.to_pandas()

# Print the entire DataFrame
print(df)

# Alternatively, if the DataFrame is very large, you might want to display only the first few rows to check its structure
print(df.head())

# Displaying DataFrame info to understand the columns and data types
print(df.info())




