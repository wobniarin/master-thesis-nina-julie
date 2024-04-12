import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import pytz
import numpy as np  # Make sure to import NumPy for sqrt function


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
    'US-TEX-ERCO': 37,  # Texas capacity in GW
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
    return df_combined

def visualize_hourly_seasonality(predicted_file, target_file, horizon, power_type='wind'):
    df_combined = split_horizon(predicted_file, target_file, horizon)
    zone = df_combined['zone_key'].iloc[0]
    
    # Calculate error
    df_combined['error'] = df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target']
    
    # Determine the correct capacity based on the power type
    capacity_gw = zone_wind_capacity_gw[zone] if power_type == 'wind' else zone_solar_capacity_gw[zone]

    df_combined['target_time'] = pd.to_datetime(df_combined['target_time'], unit='ms', utc=True)
    df_combined.set_index('target_time', inplace=True)

    # Calculate daily NMBE
    nmbe_by_day = df_combined['error'].rolling('30D').mean() / capacity_gw

    # Filter for January month of a specific year
    start_january = pd.to_datetime('2023-01-01', utc=True)
    end_january = pd.to_datetime('2023-01-31', utc=True)
    nmbe_by_january = nmbe_by_day[start_january:end_january]

    # Print NMBE for January days
    print(f"Daily NMBE for January {zone} - {power_type.capitalize()} Power, Horizon: {horizon}")
    for date, nmbe in nmbe_by_january.iteritems():
        print(f"Date {date}: {nmbe:.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(nmbe_by_january.index, nmbe_by_january, marker='o', linestyle='-', color='blue')
    plt.title(f'Rolling Daily NMBE for January {power_type.capitalize()} Power\nZone: {zone}, Horizon: {horizon}')
    plt.xlabel('Date')
    plt.ylabel('NMBE')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the visualization function
for predicted_file, target_file in target_predicted_files.items():
    visualize_hourly_seasonality(predicted_file, target_file, 24, 'wind')
