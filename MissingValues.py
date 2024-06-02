import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import pytz
import numpy as np 


# Define the file paths
target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

# Timezone mapping for each zone
timezone_mapping = {
    'US-CAL-CISO': 'America/Los_Angeles',  # California Time Zone
    'US-TEX-ERCO': 'America/Chicago',      # Texas Time Zone
}

power_types = ['wind', 'solar']

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
    start_date = pd.Timestamp('2023-02-01', tz=timezone_mapping[zone_key])
    end_date = pd.Timestamp('2024-04-30', tz=timezone_mapping[zone_key])
    df_combined = df_combined[(df_combined['target_time'] >= start_date) & (df_combined['target_time'] <= end_date)]
    return df_combined

# Loop through each file pair
for predicted_file, target_file in target_predicted_files.items():

    df_combined = split_horizon(predicted_file, target_file, 24)
    
    for power_type in power_types:
        predicted_missing = df_combined[f'power_production_{power_type}_avg_pred'].isnull().sum().sum()
        target_missing = df_combined[f'power_production_{power_type}_avg_target'].isnull().sum().sum()
    
        print('numer of missing values in predicted ' + power_type + ':' + str(predicted_missing))
        print('numer of missing values in target ' + power_type + ':' + str(target_missing))

   