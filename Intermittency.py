import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# Define the locations of your predicted and target data files
target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
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
    print(df_combined)
    return df_combined
    

def visualize_weekly_data(predicted_file, target_file, horizon, power_type='wind', week_start='2023-09-04'):
    df_combined = split_horizon(predicted_file, target_file, horizon)
    zone = df_combined['zone_key_pred'].iloc[0] 
    
    # Convert 'week_start' to a timezone-aware datetime object
    week_start_date = pd.to_datetime(week_start).tz_localize('UTC')
    one_week_later = week_start_date + pd.Timedelta(days=7)
    
    # Filter the combined dataframe for the specified week
    df_week = df_combined[(df_combined['target_time'] >= week_start_date) & (df_combined['target_time'] < one_week_later)]
    
    plt.figure(figsize=(15, 7))
    
    for column_name, color, label in [
        (f'power_production_{power_type}_avg_pred', 'blue', 'Predicted'),
        (f'power_production_{power_type}_avg_target', 'red', 'Target')
    ]:
        plt.plot(df_week['target_time'], df_week[column_name], marker='o', linestyle='-', color=color, label=label)
    
    plt.title(f'Intermittency for {power_type.capitalize()} Power\nZone: {zone}')
    plt.xlabel('Time')
    plt.ylabel('GWh')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    
    # Set the y-axis range
    plt.ylim(0, 16500)  # Adjusting y-axis to have a range up to 16500
    
    plt.tight_layout()
    plt.show()

# Loop over each pair of predicted and target files, visualizing the data
for predicted_file, target_file in target_predicted_files.items():
    visualize_weekly_data(predicted_file, target_file, 12, 'wind', week_start='2023-09-04')
    visualize_weekly_data(predicted_file, target_file, 24, 'wind', week_start='2023-09-04')
