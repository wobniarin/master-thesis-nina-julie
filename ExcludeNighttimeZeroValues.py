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


def separate_times_and_save(files):
    for predicted_path, target_path in files.items():
        predicted_df = pq.read_table(predicted_path).to_pandas()
        target_df = pq.read_table(target_path).to_pandas()

        nighttime_mask = target_df['power_production_solar_avg'] == 0
        daytime_mask = target_df['power_production_solar_avg'] > 0

        # Separate nighttime and daytime data
        df_nighttime_predicted = predicted_df[nighttime_mask]
        df_daytime_predicted = predicted_df[daytime_mask]

        df_nighttime_target = target_df[nighttime_mask]
        df_daytime_target = target_df[daytime_mask]

        # Define new file paths
        predicted_nighttime_path = predicted_path.replace(".parquet", "_nighttime.parquet")
        target_nighttime_path = target_path.replace(".parquet", "_nighttime.parquet")

        predicted_daytime_path = predicted_path.replace(".parquet", "_daytime.parquet")
        target_daytime_path = target_path.replace(".parquet", "_daytime.parquet")

        # Save the nighttime DataFrames as new Parquet files
        df_nighttime_predicted.to_parquet(predicted_nighttime_path, index=False)
        df_nighttime_target.to_parquet(target_nighttime_path, index=False)

        # Save the daytime DataFrames as new Parquet files
        df_daytime_predicted.to_parquet(predicted_daytime_path, index=False)
        df_daytime_target.to_parquet(target_daytime_path, index=False)

        # You can return the paths or just print a confirmation
        print(f"Saved nighttime data to {predicted_nighttime_path} and {target_nighttime_path}")
        print(f"Saved daytime data to {predicted_daytime_path} and {target_daytime_path}")

# Call the function
separate_times_and_save(target_predicted_files)
