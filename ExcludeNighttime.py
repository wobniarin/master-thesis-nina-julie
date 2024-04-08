from astral.sun import sun
from astral import LocationInfo
from datetime import datetime
from pytz import timezone
import pandas as pd
import pyarrow.parquet as pq


target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}


# Function to calculate the broadest daylight hours for California
def is_daytime(dt, eastern_point, western_point):
    s_east = sun(eastern_point.observer, date=dt.date(), tzinfo=timezone(eastern_point.timezone))
    s_west = sun(western_point.observer, date=dt.date(), tzinfo=timezone(western_point.timezone))
    
    # Use the earliest sunrise and the latest sunset
    sunrise = min(s_east['sunrise'], s_west['sunrise'])
    sunset = max(s_east['sunset'], s_west['sunset'])
    
    return sunrise <= dt <= sunset

"""
def exclude_nighttimes(files):
    results = {}  # Initialize a dictionary to hold the results

    for predicted_path, target_path in files.items():
        
        if "US-CAL-CISO" in predicted_path:
            # Locations representing easternmost and westernmost points in California
            eastern_point = LocationInfo("Lake Tahoe", "USA", "America/Los_Angeles", 39.0968, -120.0324)
            western_point = LocationInfo("Point Arena", "USA", "America/Los_Angeles", 38.9185, -123.7111)
        elif "US-TEX-ERCO" in predicted_path:
            # Locations representing easternmost and westernmost points in Texas
            eastern_point = LocationInfo("Orange", "USA", "America/Chicago", 30.1035, -93.7495)
            western_point = LocationInfo("El Paso", "USA", "America/Denver", 31.7619, -106.4850)

        predicted_df = pq.read_table(predicted_path).to_pandas()
        target_df = pq.read_table(target_path).to_pandas()

        predicted_df['datetime'] = pd.to_datetime(predicted_df['target_time'], unit='ms')
        predicted_df['is_daytime'] = predicted_df['datetime'].apply(lambda dt: is_daytime(dt, eastern_point, western_point))

        df_daytime_predicted = predicted_df[predicted_df['is_daytime']]
        df_daytime_predicted.set_index(['target_time'], inplace=True)
        target_df.set_index(['target_time'], inplace=True)

        df_target_daytime = target_df.loc[df_daytime_predicted.index.unique()]




        # Use the original file paths as keys in the results dictionary
        results[predicted_path] = (df_daytime_predicted, df_target_daytime)

    return results
"""     

def exclude_nighttimes_and_save(files):
    results = {}  # Initialize a dictionary to hold the new file paths

    for predicted_path, target_path in files.items():
        if "US-CAL-CISO" in predicted_path:
            eastern_point = LocationInfo("Lake Tahoe", "USA", "America/Los_Angeles", 39.0968, -120.0324)
            western_point = LocationInfo("Point Arena", "USA", "America/Los_Angeles", 38.9185, -123.7111)
        elif "US-TEX-ERCO" in predicted_path:
            eastern_point = LocationInfo("Orange", "USA", "America/Chicago", 30.1035, -93.7495)
            western_point = LocationInfo("El Paso", "USA", "America/Denver", 31.7619, -106.4850)

        predicted_df = pq.read_table(predicted_path).to_pandas()
        target_df = pq.read_table(target_path).to_pandas()

        predicted_df['datetime'] = pd.to_datetime(predicted_df['target_time'], unit='ms')
        predicted_df['is_daytime'] = predicted_df['datetime'].apply(lambda dt: is_daytime(dt, eastern_point, western_point))

        df_daytime_predicted = predicted_df[predicted_df['is_daytime']]
        df_daytime_predicted.set_index(['target_time'], inplace=True)
        target_df.set_index(['target_time'], inplace=True)

        df_target_daytime = target_df.loc[df_daytime_predicted.index.unique()]

        # Define new file paths for daytime data
        predicted_daytime_path = predicted_path.replace(".parquet", "_daytime.parquet")
        target_daytime_path = target_path.replace(".parquet", "_daytime.parquet")

        # Save the filtered DataFrames as new Parquet files
        df_daytime_predicted.reset_index().to_parquet(predicted_daytime_path, index=False)
        df_target_daytime.reset_index().to_parquet(target_daytime_path, index=False)

        # Update the results dictionary with the new paths
        results[predicted_daytime_path] = target_daytime_path

    return results


print(exclude_nighttimes_and_save(target_predicted_files))