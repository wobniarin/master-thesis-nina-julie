import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import pytz

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
    df_combined['error'] = df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target']
    df_combined['abs_error'] = df_combined['error'].abs()
    df_combined['hour'] = df_combined['target_time'].dt.hour.astype(int)
    error_by_hour = df_combined.groupby('hour')['abs_error'].mean().reindex(range(0, 24)).sort_index()

    # Plotting
    plt.figure(figsize=(12, 6))
    # Append the first hour's data to the end for a continuous plot
    if 0 in error_by_hour.index:
        error_by_hour.loc[24] = error_by_hour.loc[0]
    error_by_hour.sort_index(inplace=True)
    plt.plot(error_by_hour.index, error_by_hour, marker='o', linestyle='-', color='blue')
    plt.xticks(list(range(0, 25)))  # Set x-ticks to ensure all hours, including the appended '24', are shown
    plt.title(f'Average Hourly MAE for {power_type.capitalize()} Power\nZone: {zone}, Horizon: {horizon}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Absolute Error (kWh)')
    plt.ylim(0, 2500)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example function calls for each predicted-target file pair and power type
for predicted_file, target_file in target_predicted_files.items():
    visualize_hourly_seasonality(predicted_file, target_file, 12, 'solar')
    visualize_hourly_seasonality(predicted_file, target_file, 24, 'solar')
