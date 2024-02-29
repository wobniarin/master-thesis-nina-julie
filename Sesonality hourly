import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import os

# Dictionary mapping predicted file paths to target file paths
target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

def split_horizon(predicted_file, target_file, horizon):
    # Read parquet files and convert to pandas DataFrames
    df_predicted = pq.read_table(predicted_file).to_pandas()
    df_target = pq.read_table(target_file).to_pandas()

    # Filter DataFrames based on the specified horizon
    df_predicted = df_predicted[df_predicted["horizon"] == horizon].copy()
    df_target = df_target[df_target["horizon"] == horizon].copy()

    # Merge predicted and target DataFrames on 'target_time'
    df_predicted['target_time'] = pd.to_datetime(df_predicted['target_time'])
    df_target['target_time'] = pd.to_datetime(df_target['target_time'])
    df_combined = pd.merge(df_predicted, df_target, on='target_time', suffixes=('_pred', '_target'))

    return df_combined

def visualize_hourly_seasonality(predicted_file, target_file, horizon, power_type='wind'):
    df_predicted = pq.read_table(predicted_file).to_pandas()
    zone = df_predicted['zone_key'].iloc[0]  # Assuming 'zone_key' is present

    df_combined = split_horizon(predicted_file, target_file, horizon)
    df_combined['error'] = df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target']
    df_combined['abs_error'] = df_combined['error'].abs()
    

    # Filter data within April 2023 to January 2024 and extract hour
    df_combined['target_time'] = pd.to_datetime(df_combined['target_time'])
    df_combined['hour'] = df_combined['target_time'].dt.hour

    # Group by hour and calculate mean absolute error
    error_by_hour = df_combined.groupby('hour')['abs_error'].mean()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(error_by_hour.index, error_by_hour, marker='o', linestyle='-', color='blue')
    #zone = df_combined['zone_key'].iloc[0]
    #zone = os.path.basename(predicted_file).split('_')[1]  # Extract zone from file name
    plt.title(f'Average Hourly MAE for {power_type.capitalize()} Power\nZone: {zone}, Horizon: {horizon}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(range(0, 24))  # Ensure x-ticks for every hour
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example function calls for each predicted-target file pair and power type
for predicted_file, target_file in target_predicted_files.items():
    visualize_hourly_seasonality(predicted_file, target_file, 12, 'solar')
    visualize_hourly_seasonality(predicted_file, target_file, 24, 'solar')
