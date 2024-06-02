import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# Define the locations of your predicted and target data files
target_predicted_files_CAL = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
}

target_predicted_files_TEX = {
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

#Max capacity in MW
US_CAL_CISO_solar_capacity = 19700
US_CAL_CISO_wind_capacity = 6030
US_TEX_ERCO_solar_capacity = 13500
US_TEX_ERCO_wind_capacity = 37000

def split_horizon(predicted_file, target_file, horizon):
    # Read the data from the files
    df_predicted = pq.read_table(predicted_file).to_pandas()
    df_target = pq.read_table(target_file).to_pandas()

    # Filter data by the specified forecast horizon
    df_predicted = df_predicted[df_predicted["horizon"] == horizon].copy()
    df_target = df_target[df_target["horizon"] == horizon].copy()

    # Ensure 'target_time' is a datetime type and timezone-aware (UTC)
    df_predicted['target_time'] = pd.to_datetime(df_predicted['target_time'], utc=True)
    df_target['target_time'] = pd.to_datetime(df_target['target_time'], utc=True)

    # Merge the predicted and target dataframes on 'target_time'
    df_combined = pd.merge(df_predicted, df_target, on='target_time', suffixes=('_pred', '_target'))
    return df_combined


def visualize_data(predicted_file, target_file, horizon, power_type, capacity, zone):
    df_combined = split_horizon(predicted_file, target_file, horizon)
    #zone = df_combined['zone_key'].iloc[0] 

    plt.figure(figsize=(15, 7))
    
    # Plot predicted, target, and naive forecast data
    plot_configs = [
        (f'power_production_{power_type}_avg_pred', 'blue', 'Predicted'),
        (f'power_production_{power_type}_avg_target', 'red', 'Target'),
    ]

    for column_name, color, label in plot_configs:
        plt.plot(df_combined['target_time'], df_combined[column_name], linestyle='-', color=color, label=label)
    
    plt.title(f'Hourly {power_type.capitalize()} Power Production for {zone.capitalize()}')
    plt.ylabel('MWh')

    # Add a horizontal line at the maximum capacity
    plt.axhline(y=capacity, color='green', linestyle='--', linewidth=2, label='Max Capacity')
    plt.legend(loc='upper right')
    
    # Set the y-axis range
    plt.ylim(-10, 38000)  # Adjusting y-axis to have a range up to max capacity
    
    plt.tight_layout()
    plt.show()



for predicted_file, target_file in target_predicted_files_TEX.items():
    #visualize_data(predicted_file, target_file, 24, 'solar', capacity=US_TEX_ERCO_solar_capacity, zone='Texas')
    visualize_data(predicted_file, target_file, 24, 'wind', capacity=US_TEX_ERCO_wind_capacity, zone='Texas')
"""

for predicted_file, target_file in target_predicted_files_CAL.items():
    #visualize_data(predicted_file, target_file, 24, 'solar', capacity=US_CAL_CISO_solar_capacity, zone='California')
    visualize_data(predicted_file, target_file, 24, 'wind', capacity=US_CAL_CISO_wind_capacity, zone='California')
"""