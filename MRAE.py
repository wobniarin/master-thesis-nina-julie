import pandas as pd
import pytz
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import matplotlib.dates as mdates

# Dictionary mapping predicted file paths to target file paths
target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

# Timezone mapping for each zone
timezone_mapping = {
    'US-CAL-CISO': 'America/Los_Angeles',
    'US-TEX-ERCO': 'America/Chicago',
}

# Capacities in MW
zone_capacity_mw = {
    'US-CAL-CISO': {'solar': 19700, 'wind': 6030},
    'US-TEX-ERCO': {'solar': 13500, 'wind': 37000},
}

naive_CAL = 'naive_forecast_US-CAL-CISO.parquet'
naive_TEX = 'naive_forecast_US-TEX-ERCO.parquet'

#Returning a combined dataframe with predicted and target for the chosen horizon
def split_horizon(df_predicted, df_target, df_naive, zone_key, horizon):
    
    #convert to local time
    df_predicted = convert_to_local_time(df_predicted, zone_key)
    df_target = convert_to_local_time(df_target, zone_key)
    df_naive = convert_to_local_time(df_naive, zone_key)

    #split by horizon (24 or 12)
    df_predicted = df_predicted[df_predicted["horizon"] == horizon].copy()
    df_target = df_target[df_target["horizon"] == horizon].copy()
    df_naive = df_naive[df_naive["horizon"] == horizon].copy()
    
    #combine the three dataframes, predicted, target, and naive
    df_combined = pd.merge(df_predicted, df_target, on='target_time', suffixes=('_pred', '_target'))
    df_combined = pd.merge(df_naive, df_combined, on='target_time')
    
    #return the combined datafram
    return df_combined

#Returning a new combined dataframe for the chosen timeframe
def split_time(df_combined, start, end, zone_key):
    start_date = pd.Timestamp(start, tz=timezone_mapping[zone_key])
    end_date = pd.Timestamp(end, tz=timezone_mapping[zone_key])
    df_combined = df_combined[(df_combined['target_time'] >= start_date) & (df_combined['target_time'] <= end_date)]
    return df_combined

#Converting target_time from ms to dates
def convert_to_local_time(df, zone_key):
    df['target_time'] = pd.to_datetime(df['target_time'], unit='ms', utc=True)
    local_timezone = pytz.timezone(timezone_mapping[zone_key])
    df['target_time'] = df['target_time'].dt.tz_convert(local_timezone)
    return df

def calculate_daily_mrae(df, column_pred, column_naive, column_target):
    # Calculate daily MAE for predictive and naive models
    df['mae_pred'] = np.abs(df[column_pred] - df[column_target])
    df['mae_naive'] = np.abs(df[column_naive] - df[column_target])

    # Resample the errors daily and calculate daily MRAE
    daily_mae_pred = df['mae_pred'].resample('D').mean()
    daily_mae_naive = df['mae_naive'].resample('D').mean()
    
    daily_mrae = daily_mae_pred / daily_mae_naive

    return daily_mrae

def visualize_daily_mrae(predicted_file, target_file):
    # Load the data
    df_predicted = pq.read_table(predicted_file).to_pandas()
    df_target = pq.read_table(target_file).to_pandas()

    #extracts the zone from one of the DataFrames
    zone_key = df_predicted['zone_key'].iloc[0]
    if zone_key == 'US-CAL-CISO':
        zone_name = 'California'
    else:
        zone_name = 'Texas'

    power_type = 'wind'  # This can be set to wind/solar

    #finding the right max capacity
    capacity_mw = zone_capacity_mw[zone_key][power_type]

    if zone_key == 'US-CAL-CISO':
        df_naive = pq.read_table(naive_CAL).to_pandas()
    else:
        df_naive = pq.read_table(naive_TEX).to_pandas()

    # Split data by horizon, focusing on horizon 24 and discarding horizon 12
    df_combined = split_horizon(df_predicted, df_target, df_naive, zone_key, 24)

    #Split data by time
    df_combined = split_time(df_combined, start='2023-08-01', end='2023-08-14', zone_key=zone_key)
    
    #Set index
    df_combined.set_index('target_time', inplace=True)

    # Calculate daily MRAE
    daily_mrae = calculate_daily_mrae(df_combined, f'power_production_{power_type}_avg_pred', f'naive_forecast_{power_type}', f'power_production_{power_type}_avg_target')

    # Plotting daily NMAE
    plt.figure(figsize=(12, 6))
    plt.plot(daily_mrae.index, daily_mrae, linestyle='-', marker='o', color='blue', label='Daily MRAE')
    plt.title(f'Daily MRAE for {zone_name} - {power_type.capitalize()} Power Production')
    plt.xlabel('Date')
    plt.ylabel('Logarithmic MRAE')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    # Set x-ticks to be every day and format the dates
    plt.grid(True)
    plt.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Line of equal performance between the two models')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Call the visualization function
for predicted_file, target_file in target_predicted_files.items():
    visualize_daily_mrae(predicted_file, target_file)
