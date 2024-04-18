import pandas as pd
import pytz
import matplotlib.pyplot as plt
import numpy as np

# Path to your Parquet files
target_files = {
    'US-CAL-CISO': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'US-TEX-ERCO': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

timezone_mapping = {
    'US-CAL-CISO': 'America/Los_Angeles',  # California Time Zone
    'US-TEX-ERCO': 'America/Chicago',      # Texas Time Zone
}

# Wind capacities in GW, to be converted to MW for calculation
zone_wind_capacity_gw = {
    'US-CAL-CISO': 6.03,
    'US-TEX-ERCO': 37,
}

def convert_to_local_time(df, zone_key):
    df['target_time'] = pd.to_datetime(df['target_time'], unit='ms', utc=True)
    local_timezone = pytz.timezone(timezone_mapping[zone_key])
    df['target_time'] = df['target_time'].dt.tz_convert(local_timezone)
    return df

# Define the specific period
start_date = pd.Timestamp('2023-08-01', tz='UTC')
end_date = pd.Timestamp('2023-08-14', tz='UTC')

# Process each file
for zone_key, file_path in target_files.items():
    # Load the dataset
    df = pd.read_parquet(file_path)
    
    # Convert target_time to the local timezone for the region
    df = convert_to_local_time(df, zone_key)
    
    # Filter rows for the defined period
    df_period = df[(df['target_time'] >= start_date.tz_convert(timezone_mapping[zone_key])) & 
                   (df['target_time'] <= end_date.tz_convert(timezone_mapping[zone_key]))]
    
    # For naive forecast, shift the values by 48 hours
    df_period = df_period.sort_values('target_time')
    df_period['naive_forecast'] = df_period['power_production_wind_avg'].shift(48)
    
    # Calculate MAE and normalize by wind capacity in MW
    capacity_mw = zone_wind_capacity_gw[zone_key] * 1000  # Convert GW to MW
    df_period['mae'] = (df_period['power_production_wind_avg'] - df_period['naive_forecast']).abs()
    normalized_mae = df_period['mae'].mean() / capacity_mw
    print(f"Normalized Mean Absolute Error for {zone_key} in MW: {normalized_mae:.2f}")
    
    # Plotting normalized MAE in MW
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_period['target_time'], df_period['mae'] / capacity_mw, label=f"Wind", color='blue')
    ax.set_title(f'Normalized MAE (Wind) for {zone_key}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized MAE per MW')
    ax.legend()
    plt.show()
    
    # Save the result to a new file or handle it as needed
    output_file_path = f'naive_forecast_{zone_key}.parquet'
    df_period.to_parquet(output_file_path, index=False)

