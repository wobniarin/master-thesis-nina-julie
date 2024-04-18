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

def visualize_weekly_data(predicted_file, target_file, horizon, power_type, week_start, week_end, capacity, zone):
    df_combined = split_horizon(predicted_file, target_file, horizon)
    #zone = df_combined['zone_key'].iloc[0] 
    
    # Convert 'week_start' to a timezone-aware datetime object
    week_start_date = pd.to_datetime(week_start).tz_localize('UTC')
    week_end_date = pd.to_datetime(week_end).tz_localize('UTC')
    
    # Filter the combined dataframe for the specified week
    df_week = df_combined[(df_combined['target_time'] >= week_start_date) & (df_combined['target_time'] < week_end_date)]
    
    plt.figure(figsize=(15, 7))
    
    for column_name, color, label in [
        (f'power_production_{power_type}_avg_pred', 'blue', 'Predicted'),
        (f'power_production_{power_type}_avg_target', 'red', 'Target')
    ]:
        plt.plot(df_week['target_time'], df_week[column_name], linestyle='-', color=color, label=label)
    
    plt.title(f'Hourly {power_type.capitalize()} Power Production for {zone.capitalize()}')
    plt.xlabel('Time')
    plt.ylabel('MWh')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    # Add a horizontal line at the maximum capacity
    plt.axhline(y=capacity, color='green', linestyle='--', linewidth=2, label='Max Capacity')
    
    # Set the y-axis range
    plt.ylim(-10, 10000)  # Adjusting y-axis to have a range up to 16500
    
    plt.tight_layout()
    plt.show()

# Loop over each pair of predicted and target files, visualizing the data

US_CAL_CISO_solar_capacity = 19700
US_CAL_CISO_wind_capacity = 6030
US_TEX_ERCO_solar_capacity = 13500
US_TEX_ERCO_wind_capacity = 37000


for predicted_file, target_file in target_predicted_files_CAL.items():
    #visualize_weekly_data(predicted_file, target_file, 24, 'solar', week_start='2023-08-01', week_end='2023-08-14', capacity=US_CAL_CISO_solar_capacity, zone='California')
    visualize_weekly_data(predicted_file, target_file, 24, 'wind', week_start='2023-08-01', week_end='2023-08-14', capacity=US_CAL_CISO_wind_capacity, zone='California')

"""
for predicted_file, target_file in target_predicted_files_TEX.items():
    visualize_weekly_data(predicted_file, target_file, 24, 'solar', week_start='2023-08-01', week_end='2023-08-14', capacity=US_TEX_ERCO_solar_capacity, zone='Texas')
    visualize_weekly_data(predicted_file, target_file, 24, 'wind', week_start='2023-08-01', week_end='2023-08-14', capacity=US_TEX_ERCO_wind_capacity, zone='Texas')
"""