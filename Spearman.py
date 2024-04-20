import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import pytz
import numpy as np 
from scipy.stats import spearmanr
import matplotlib.dates as mdates

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
    'US-CAL-CISO': 19700,
    'US-TEX-ERCO': 13500,
}

zone_wind_capacity_gw = {
    'US-CAL-CISO': 6030,
    'US-TEX-ERCO': 37000,
}


def convert_to_local_time(df, zone_key):
    #This line converts the timestamps in the 'target_time' column to datetime objects.
    df['target_time'] = pd.to_datetime(df['target_time'], unit='ms', utc=True)

    #Retrieves the local time zone corresponding to the zone_key 
    local_timezone = pytz.timezone(timezone_mapping[zone_key])

    #converts the datetime objects in the 'target_time' column from UTC time zone to the local time zone
    df['target_time'] = df['target_time'].dt.tz_convert(local_timezone)
    
    return df


def split_horizon(predicted_file, target_file, horizon):
    #Convert to pandas DataFrames
    df_predicted = pq.read_table(predicted_file).to_pandas()
    df_target = pq.read_table(target_file).to_pandas()

    #Extracts the zone_key
    zone_key = df_predicted['zone_key'].iloc[0]

    #converts the timestamp columns ('target_time') to the local time zone
    df_predicted = convert_to_local_time(df_predicted, zone_key)
    df_target = convert_to_local_time(df_target, zone_key)

    #filters DataFrames to include only the rows where the 'horizon' column matches the specified horizon (24 or 12)
    df_predicted = df_predicted[df_predicted["horizon"] == horizon].copy()
    df_target = df_target[df_target["horizon"] == horizon].copy()

    #Merges the filtered DataFrames on the 'target_time' column, appending suffixes to column names to differentiate between predicted and target values.
    df_combined = pd.merge(df_predicted, df_target, on='target_time', suffixes=('_pred', '_target'))
    
    #defines a time range from start_date to end_date, both inclusive, in the local time zone based on the zone_key
    start_date = pd.Timestamp('2023-08-01', tz=timezone_mapping[zone_key])
    end_date = pd.Timestamp('2023-08-14', tz=timezone_mapping[zone_key])

    #filters the combined DataFrame to include only rows where the 'target_time' falls within the specified time range.
    df_combined = df_combined[(df_combined['target_time'] >= start_date) & (df_combined['target_time'] <= end_date)]

    return df_combined

def visualize_daily_spearman(predicted_file, target_file, horizon, power_type='solar'):
    
    #calls the split_horizon function to obtain a new combined dataframe with specified horizon
    df_combined = split_horizon(predicted_file, target_file, horizon) 

    #extracts the zone from the df_combined DataFrame
    zone = df_combined['zone_key_pred'].iloc[0]

    #sets 'target_time' as the index for dataframe
    df_combined.set_index('target_time', inplace=True)

    # Calculate absolute error in MW, between the predicted and target power production values for each row and adds it as a new column named 'abs_error'
    df_combined['abs_error'] = np.abs(df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])

   
    # Prepare daily data without aggregating into a single mean or sum
    df_daily = df_combined.resample('D')

    daily_spearman = []
    # Iterating over each day as a group
    for date, group in df_daily:
        if len(group) > 1:  # Ensure there's enough data to compute Spearman
            spearman_corr, _ = spearmanr(group[f'power_production_{power_type}_avg_target'], group['abs_error'])
            daily_spearman.append(spearman_corr)
        else:
            daily_spearman.append(np.nan)  # Append NaN if not enough data
    
        
    # Plotting daily Spearman
    plt.figure(figsize=(12, 6))
    plt.plot([date for date, _ in df_daily], daily_spearman, linestyle='-', marker='o', color='green', markersize=5, label='Daily Spearman Rank Correlation')
    #plt.plot(df_daily.index, daily_spearman, linestyle='-', marker='o', color='green', markersize=5, label='Daily Spearman Rank Correlation')
    plt.title(f'Daily Spearman Rank Correlation for {zone} - {power_type.capitalize()} Power Production')
    plt.xlabel('Date')
    plt.ylabel('Spearman Rank Correlation')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    plt.show()

# Call the visualization function
for predicted_file, target_file in target_predicted_files.items():
    visualize_daily_spearman(predicted_file, target_file, 24, 'solar')  