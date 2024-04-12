import pandas as pd
import pytz

# The last available target time in milliseconds
last_target_time_ms = 1705453200000
# Convert to a timezone-aware UTC datetime object
last_target_time = pd.to_datetime(last_target_time_ms, unit='ms', utc=True)

# Define the start of the last week
one_week = pd.Timedelta(days=7)
start_time = last_target_time - one_week

# Path to your Parquet files
target_files = {
    'US-CAL-CISO': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'US-TEX-ERCO': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

timezone_mapping = {
    'US-CAL-CISO': 'America/Los_Angeles',  # California Time Zone
    'US-TEX-ERCO': 'America/Chicago',      # Texas Time Zone
}

def convert_to_local_time(df, zone_key):
    # Convert target_time to UTC then to the local timezone for the region
    df['target_time'] = pd.to_datetime(df['target_time'], unit='ms', utc=True)
    local_timezone = pytz.timezone(timezone_mapping[zone_key])
    df['target_time'] = df['target_time'].dt.tz_convert(local_timezone)
    return df

# Process each file
for zone_key, file_path in target_files.items():
    # Load the dataset
    df = pd.read_parquet(file_path)
    
    # Convert target_time to the local timezone for the region
    df = convert_to_local_time(df, zone_key)
    
    # Filter rows for the last week
    # Ensure comparison is between timezone-aware datetimes
    df_last_week = df[df['target_time'] > start_time.tz_convert(timezone_mapping[zone_key])]
    
    # For naive forecast, shift the values by 24 hours
    # Ensure sorting by target_time before shifting
    df_last_week = df_last_week.sort_values('target_time')
    df_last_week['naive_forecast'] = df_last_week.groupby(['zone_key'])['power_production_wind_avg'].shift(24)
    
    # Save the result to a new file or handle it as needed
    output_file_path = f'naive_forecast_{zone_key}.parquet'
    df_last_week.to_parquet(output_file_path, index=False)
    print(f"Naive forecast for {zone_key} saved to {output_file_path}")
