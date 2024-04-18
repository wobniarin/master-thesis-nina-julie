import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import pytz
import numpy as np  # Required for numerical operations

# Dictionary mapping predicted file paths to target file paths
target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

# Timezone mapping for each zone
timezone_mapping = {
    'US-CAL-CISO': 'America/Los_Angeles',  # California Time Zone
    'US-TEX-ERCO': 'America/Chicago',      # Texas Time Zone
}

zone_solar_capacity_gw = {
    'US-CAL-CISO': 19.7,  # California capacity in GW
    'US-TEX-ERCO': 13.5,  # Texas capacity in GW
}

zone_wind_capacity_gw = {
    'US-CAL-CISO': 6.03,  # California capacity in GW
    'US-TEX-ERCO': 37,    # Texas capacity in GW
}

def convert_to_local_time(df, zone_key):
    df['target_time'] = pd.to_datetime(df['target_time'], unit='ms', utc=True)
    local_timezone = pytz.timezone(timezone_mapping[zone_key])
    df['target_time'] = df['target_time'].dt.tz_convert(local_timezone)
    return df

def split_horizon(predicted_file, target_file, horizon):
    df_predicted = pq.read_table(predicted_file).to_pandas()
    df_target = pq.read_table(target_file).to_pandas()
    zone_key = df_predicted['zone_key'].iloc[0]
    df_predicted = convert_to_local_time(df_predicted, zone_key)
    df_target = convert_to_local_time(df_target, zone_key)
    df_predicted = df_predicted[df_predicted["horizon"] == horizon].copy()
    df_target = df_target[df_target["horizon"] == horizon].copy()
    df_combined = pd.merge(df_predicted, df_target, on='target_time', suffixes=('_pred', '_target'))

    # Calculate the last available target_time and start of last week
    last_target_time = df_combined['target_time'].max()
    one_week = pd.Timedelta(days=7)
    start_time = last_target_time - one_week

    # Filter df_combined for the last week
    df_combined = df_combined[df_combined['target_time'] >= start_time]

    return df_combined

def visualize_hourly_seasonality(predicted_file, target_file, horizon, power_type='wind'):
    df_combined = split_horizon(predicted_file, target_file, horizon)
    zone = df_combined['zone_key_pred'].iloc[0]
    
    # Calculate error
    df_combined['error'] = df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target']
    
    # Determine the correct capacity based on the power type
    capacity_gw = zone_wind_capacity_gw[zone] if power_type == 'wind' else zone_solar_capacity_gw[zone]

    df_combined.set_index('target_time', inplace=True)

    # Calculate daily NMBE
    nmbe_by_day = df_combined['error'].rolling('30D').mean() / capacity_gw

    # Plotting both NMBE and power production
    plt.figure(figsize=(12, 6))
    plt.plot(nmbe_by_day.index, nmbe_by_day, linestyle='-', color='blue', label='NMBE')
    plt.plot(df_combined.index, df_combined[f'power_production_{power_type}_avg_pred'], linestyle='-', color='green', label='Predicted Power')
    plt.plot(df_combined.index, df_combined[f'power_production_{power_type}_avg_target'], linestyle='-', color='red', label='Target Power')
    
    plt.title(f'Rolling Daily NMBE and Power Production\nZone: {zone}, Horizon: {horizon}, Power Type: {power_type.capitalize()}')
    plt.xlabel('Date')
    plt.ylabel('NMBE / Power Production (MW)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the visualization function
for predicted_file, target_file in target_predicted_files.items():
    visualize_hourly_seasonality(predicted_file, target_file, 24, 'wind')
    visualize_hourly_seasonality(predicted_file, target_file, 12, 'wind')

