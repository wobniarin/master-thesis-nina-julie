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
    start_date = pd.Timestamp('2023-08-01', tz=timezone_mapping[zone_key])
    end_date = pd.Timestamp('2023-08-14', tz=timezone_mapping[zone_key])
    df_combined = df_combined[(df_combined['target_time'] >= start_date) & (df_combined['target_time'] <= end_date)]
    return df_combined

def visualize_daily_nrmse(predicted_file, target_file, horizon, power_type='wind'):
    df_combined = split_horizon(predicted_file, target_file, horizon)
    zone = df_combined['zone_key_pred'].iloc[0]
    if zone == 'US-CAL-CISO':
        zone_name = 'California'
    else:
        zone_name = 'Texas'

    if not pd.api.types.is_datetime64_any_dtype(df_combined.index):
        df_combined['target_time'] = pd.to_datetime(df_combined['target_time'], unit='ms', utc=True)
        df_combined.set_index('target_time', inplace=True)

    capacity_mw = (zone_wind_capacity_gw[zone] if power_type == 'wind' else zone_solar_capacity_gw[zone]) * 1000
    df_combined['error'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
    df_combined['squared_error'] = df_combined['error'] ** 2

    # Calculate daily NRMdSE normalized by capacity in MW
    daily_nrmse = np.sqrt(df_combined['squared_error'].resample('D').median()) / capacity_mw

    # Plotting daily NRMdSE
    plt.figure(figsize=(12, 6))
    plt.plot(daily_nrmse.index, daily_nrmse, linestyle='-', marker='o', color='red', label='Daily NRMdSE')
    plt.title(f'Daily NRMdSE for {zone_name} - {power_type.capitalize()} Power Production')
    plt.xlabel('Date')
    plt.ylabel('NRMdSE (Normalized by Capacity in MW)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the visualization function
for predicted_file, target_file in target_predicted_files.items():
    visualize_daily_nrmse(predicted_file, target_file, 24, 'solar')
    visualize_daily_nrmse(predicted_file, target_file, 24, 'wind')