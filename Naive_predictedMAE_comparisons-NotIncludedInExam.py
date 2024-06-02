import pandas as pd
import pytz
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

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

# Capacities in GW, will be converted to MW for calculations
zone_capacity_gw = {
    'US-CAL-CISO': {'solar': 19700, 'wind': 6030},
    'US-TEX-ERCO': {'solar': 13500, 'wind': 37000},
}

def split_horizon(df_predicted, df_target, zone_key, horizon):
    df_predicted = convert_to_local_time(df_predicted, zone_key)
    df_target = convert_to_local_time(df_target, zone_key)
    df_predicted = df_predicted[df_predicted["horizon"] == horizon].copy()
    df_target = df_target[df_target["horizon"] == horizon].copy()
    df_combined = pd.merge(df_predicted, df_target, on='target_time', suffixes=('_pred', '_target'))
    start_date = pd.Timestamp('2023-08-01', tz=timezone_mapping[zone_key])
    end_date = pd.Timestamp('2023-08-14', tz=timezone_mapping[zone_key])
    df_combined = df_combined[(df_combined['target_time'] >= start_date) & (df_combined['target_time'] <= end_date)]
    return df_combined

def convert_to_local_time(df, zone_key):
    df['target_time'] = pd.to_datetime(df['target_time'], unit='ms', utc=True)
    local_timezone = pytz.timezone(timezone_mapping[zone_key])
    df['target_time'] = df['target_time'].dt.tz_convert(local_timezone)
    return df

def calculate_nmae(df, column_pred, column_actual, capacity_mw):
    df['abs_error'] = np.abs(df[column_pred] - df[column_actual])
    daily_nmae = df['abs_error'].resample('D').mean() / capacity_mw
    return daily_nmae

def calculate_mrae(df, column_pred, column_actual):
    # Exclude zero actuals to avoid division by zero
    non_zero_actuals = df[df[column_actual] != 0]
    # Calculate relative absolute errors
    relative_errors = np.abs(non_zero_actuals[column_pred] - non_zero_actuals[column_actual]) / non_zero_actuals[column_actual]
    # Mean of relative absolute errors
    mrae = relative_errors.mean()
    return mrae

# Prepare data and calculate NMAE and MRAE for each model
results = []
for predicted_file, target_file in target_predicted_files.items():
    # Load the data
    df_predicted = pq.read_table(predicted_file).to_pandas()
    df_actual = pq.read_table(target_file).to_pandas()
    zone_key = df_predicted['zone_key'].iloc[0]
    power_type = 'solar'  # This can be set as needed

    # Split data by horizon, focusing on horizon 24 and discarding horizon 12
    df_combined = split_horizon(df_predicted, df_actual, zone_key, 24)

    df_predicted = convert_to_local_time(df_predicted, zone_key)
    df_actual = convert_to_local_time(df_actual, zone_key)
    df_combined = pd.merge(df_predicted, df_actual, on='target_time', suffixes=('_pred', '_target'))
    df_combined.set_index('target_time', inplace=True)

    # Set up naive forecast by shifting the actual values
    df_combined['naive_forecast'] = df_combined[f'power_production_{power_type}_avg_target'].shift(48)

    # Calculate NMAE for predictive and naive models
    capacity_mw = zone_capacity_gw[zone_key][power_type]
    nmae_predicted = calculate_nmae(df_combined, f'power_production_{power_type}_avg_pred', f'power_production_{power_type}_avg_target', capacity_mw)
    nmae_naive = calculate_nmae(df_combined, 'naive_forecast', f'power_production_{power_type}_avg_target', capacity_mw)

    # Calculate MRAE for predictive model
    mrae_predicted = calculate_mrae(df_combined, f'power_production_{power_type}_avg_pred', f'power_production_{power_type}_avg_target')

    # Print MRAE results
    print(f'MRAE for {zone_key} - {power_type.capitalize()} Power: {mrae_predicted:.4f}')

    results.append((zone_key, nmae_predicted, nmae_naive, mrae_predicted))

# Plotting NMAE results
for zone, nmae_pred, nmae_naive, _ in results:
    plt.figure(figsize=(12, 6))
    plt.plot(nmae_pred.index, nmae_pred, linestyle='-', label='Predictive NMAE')
    plt.plot(nmae_naive.index, nmae_naive, linestyle='-', label='Naive NMAE')
    plt.title(f'Comparison of Predictive and Naive NMAE for {zone} - {power_type.capitalize()} Power')
    plt.xlabel('Date')
    plt.ylabel('NMAE (Normalized by Capacity in MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
