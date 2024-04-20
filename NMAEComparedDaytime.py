import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import pytz
import numpy as np 

# Dictionary mapping predicted file paths to target file paths
target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

target_predicted_files_daytime = {
    'data/target_and_predicted/US-CAL-CISO_predicted_daytime.parquet': 'data/target_and_predicted/US-CAL-CISO_target_daytime.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted_daytime.parquet': 'data/target_and_predicted/US-TEX-ERCO_target_daytime.parquet',
}

# Timezone mapping for each zone
timezone_mapping = {
    'US-CAL-CISO': 'America/Los_Angeles',  # California Time Zone
    'US-TEX-ERCO': 'America/Chicago',      # Texas Time Zone
}

zone_solar_capacity_gw = {
    'US-CAL-CISO': 19.7,
    'US-TEX-ERCO': 13.5,
}

zone_wind_capacity_gw = {
    'US-CAL-CISO': 6.03,
    'US-TEX-ERCO': 37,
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



def visualize_combined_nmae(predicted_file, target_file, predicted_daytime_file, target_daytime_file, horizon, power_type='solar', zone='US-CAL-CISO'):
    plt.figure(figsize=(12, 6))

    # Common setup for both datasets
    start_date = pd.Timestamp('2023-08-01', tz=timezone_mapping[zone])
    end_date = pd.Timestamp('2023-08-14', tz=timezone_mapping[zone])
    capacity_mw = (zone_wind_capacity_gw[zone] if power_type == 'wind' else zone_solar_capacity_gw[zone]) * 1000

    # Helper function to plot data
    def plot_data(predicted_file, target_file, label, linestyle):
        df_combined = split_horizon(predicted_file, target_file, horizon)
        df_combined['target_time'] = pd.to_datetime(df_combined['target_time'], utc=True)
        df_combined.set_index('target_time', inplace=True)
        df_combined['abs_error'] = np.abs(df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        daily_nmae = df_combined['abs_error'].resample('D').mean() / capacity_mw
        plt.plot(daily_nmae.index, daily_nmae, linestyle=linestyle, marker='o', label=label)

    # Plot for all-day data
    plot_data(predicted_file, target_file, 'Daily NMAE All Day', '-')

    # Plot for daytime data
    plot_data(predicted_daytime_file, target_daytime_file, 'Daily NMAE Daytime', '--')

    # Finalize plot
    plt.title(f'NMAE for {zone} - {power_type.capitalize()} Power Production')
    plt.xlabel('Date')
    plt.ylabel('NMAE (Normalized by Capacity in MW)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the visualization function for each zone with corresponding files
visualize_combined_nmae(
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet',
    'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-CAL-CISO_predicted_daytime.parquet',
    'data/target_and_predicted/US-CAL-CISO_target_daytime.parquet',
    24, 'solar', 'US-CAL-CISO'
)

visualize_combined_nmae(
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet',
    'data/target_and_predicted/US-TEX-ERCO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted_daytime.parquet',
    'data/target_and_predicted/US-TEX-ERCO_target_daytime.parquet',
    24, 'solar', 'US-TEX-ERCO'
)

