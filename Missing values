import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# Define the file paths
target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

# Initialize a dictionary to hold the results
missing_values_summary = {}

# Loop through each file pair
for predicted_file, target_file in target_predicted_files.items():
    # Load the predicted and target data
    df_predicted = pq.read_table(predicted_file).to_pandas()
    df_target = pq.read_table(target_file).to_pandas()
    
    # Calculate the number of missing values in each DataFrame
    predicted_missing = df_predicted.isnull().sum().sum()
    target_missing = df_target.isnull().sum().sum()
    
    # Store the results
    missing_values_summary[predicted_file] = predicted_missing
    missing_values_summary[target_file] = target_missing

# Visualization
# Convert the summary to a DataFrame for easier plotting
missing_df = pd.DataFrame(list(missing_values_summary.items()), columns=['File', 'Missing Values'])

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(missing_df['File'], missing_df['Missing Values'], color='skyblue')
plt.xlabel('Number of Missing Values')
plt.ylabel('Files')
plt.title('Missing Values in Predicted and Target Files')
plt.tight_layout()
plt.show()