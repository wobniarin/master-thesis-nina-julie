import pandas as pd
import pytz

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

# Define the specific period
start_date = pd.Timestamp('2023-07-30', tz='UTC')
end_date = pd.Timestamp('2023-08-14', tz='UTC')

#Assume the frequency of data recording is every half hour
shift_periods = 96  # 48 hours * 2 half-hours per hour

# Process each file
for zone_key, file_path in target_files.items():
    # Load the dataset
    df = pd.read_parquet(file_path)
    
    # Convert target_time to the local timezone for the region
    df = convert_to_local_time(df, zone_key)
    
    # Filter rows for the defined period
    # Ensure comparison is between timezone-aware datetimes converted to the right time zone
    df_period = df[(df['target_time'] >= start_date.tz_convert(timezone_mapping[zone_key])) & 
                   (df['target_time'] <= end_date.tz_convert(timezone_mapping[zone_key]))]
    
    # For naive forecast, shift the values by the correct number of periods
    # Ensure sorting by target_time before shifting
    df_period = df_period.sort_values('target_time')
    df_period['naive_forecast'] = df_period.groupby(['zone_key'])['power_production_wind_avg'].shift(shift_periods)
    
    # Save the result to a new file or handle it as needed
    output_file_path = f'naive_forecast_{zone_key}.parquet'
    df_period.to_parquet(output_file_path, index=False)
    print(f"Naive forecast for {zone_key} saved to {output_file_path}")


