import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

def split_horizon(predicted_file, target_file, horizon):
    table = pq.read_table(predicted_file)
    df_predicted = table.to_pandas()
    df_predicted = df_predicted[df_predicted["horizon"] == horizon].copy()

    table = pq.read_table(target_file)
    df_target = table.to_pandas()
    df_target = df_target[df_target["horizon"] == horizon].copy()

    return df_predicted, df_target


def plot(x_value, y_value):
    max_y = max(y_value)
    max_x = max(x_value)
    maximum_y_or_x = max(max_y, max_x)
    plt.plot([0, maximum_y_or_x], [0, maximum_y_or_x], color='red')
    plt.scatter(x_value, y_value, s=10)
    plt.ylabel('Target')
    plt.xlabel('Predicted')
    print(plt.show())


def scatter_wind(predicted_file, target_file):
    predicted_file = predicted_file.copy()
    target_file = target_file.copy()

    # Set "target_time" as index for both DataFrames
    predicted_file.set_index(['target_time'], inplace=True)
    target_file.set_index(['target_time'], inplace=True)
    
    df_predicted_wind = predicted_file.dropna(subset=["power_production_wind_avg"])
    df_target_wind = target_file.dropna(subset=["power_production_wind_avg"])
    
    # Reindex or align one DataFrame to match the other
    df_target_wind = df_target_wind.reindex(df_predicted_wind.index)
    
    x = df_predicted_wind["power_production_wind_avg"].values
    y = df_target_wind["power_production_wind_avg"].values
    plot(x,y)


def scatter_solar(predicted_file, target_file):
    predicted_file = predicted_file.copy()
    target_file = target_file.copy()

    # Set "target_time" as index for both DataFrames
    predicted_file.set_index(['target_time'], inplace=True)
    target_file.set_index(['target_time'], inplace=True)

    df_predicted_solar = predicted_file.dropna(subset=["power_production_solar_avg"])
    df_target_solar = target_file.dropna(subset=["power_production_solar_avg"])

    # Reindex or align one DataFrame to match the other
    df_target_solar = df_target_solar.reindex(df_predicted_solar.index)

    x = df_predicted_solar["power_production_solar_avg"].values
    y = df_target_solar["power_production_solar_avg"].values
    plot(x,y)

if __name__ == "__main__": #So code only runs when in this file, not when imported.
    for predicted_file, target_file in target_predicted_files.items():
        df_predicted_12, df_target_12 = split_horizon(predicted_file, target_file, 12)
        scatter_wind(df_predicted_12, df_target_12)
        scatter_solar(df_predicted_12, df_target_12)

        df_predicted_24, df_target_24 = split_horizon(predicted_file, target_file, 24)
        scatter_wind(df_predicted_24, df_target_24)
        scatter_solar(df_predicted_24, df_target_24)



