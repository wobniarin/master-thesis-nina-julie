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
    df_predicted = pq.read_table(predicted_file).to_pandas()
    zone = df_predicted['zone_key'].iloc[0]
    df_combined = split_horizon(predicted_file, target_file, horizon)
    
    # Calculate squared error
    df_combined['squared_error'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target']) ** 2
    
    # Determine the correct capacity based on the power type
    if power_type == 'wind':
        capacity_gw = zone_wind_capacity_gw[zone]
    else:  # Assuming solar if not wind
        capacity_gw = zone_solar_capacity_gw[zone]

    df_combined['hour'] = df_combined['target_time'].dt.hour.astype(int)
    # Calculate Root Median Squared Error (RMSdE) for each hour, then divide by capacity
    rmsde_by_hour = df_combined.groupby('hour')['squared_error'].median().apply(np.sqrt) / capacity_gw
    rmsde_by_hour = rmsde_by_hour.reindex(range(0, 24)).sort_index()

    # Output the RMSdE divided by capacity for each hour
    print(f"RMSdE/Capacity by Hour for {zone} - {power_type.capitalize()} Power, Horizon: {horizon}")
    for hour, rmsde in rmsde_by_hour.items():
        print(f"Hour {hour}: {rmsde:.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    if 0 in rmsde_by_hour.index:
        rmsde_by_hour.loc[24] = rmsde_by_hour.loc[0]
    rmsde_by_hour.sort_index(inplace=True)
    plt.plot(rmsde_by_hour.index, rmsde_by_hour, marker='o', linestyle='-', color='blue')
    plt.xticks(list(range(0, 25)))
    plt.title(f'Average Hourly RMSdE Divided by Capacity for {power_type.capitalize()} Power\nZone: {zone}, Horizon: {horizon}')
    plt.xlabel('Hour of Day')
    plt.ylabel('RMSdE Divided by Capacity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example function calls for each predicted-target file pair and power type
for predicted_file, target_file in target_predicted_files.items():
    visualize_hourly_seasonality(predicted_file, target_file, 12, 'wind')
    visualize_hourly_seasonality(predicted_file, target_file, 24, 'wind')