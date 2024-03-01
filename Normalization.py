from Scatterplot import split_horizon, plot, scatter_wind, scatter_solar
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.colors as mcolors 


target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

for predicted_file, target_file in target_predicted_files.items():
    df_predicted_12, df_target_12 = split_horizon(predicted_file, target_file, 12)

    # Set "target_time" as index for both DataFrames
    df_predicted_12.set_index(['target_time'], inplace=True)
    df_target_12.set_index(['target_time'], inplace=True)
    
    df_predicted_wind = df_predicted_12.dropna(subset=["power_production_wind_avg"])
    df_target_wind = df_target_12.dropna(subset=["power_production_wind_avg"])
    
    # Reindex or align one DataFrame to match the other
    df_target_wind = df_target_wind.reindex(df_predicted_wind.index)
    
    x = df_predicted_wind["power_production_wind_avg"].values
    y = df_target_wind["power_production_wind_avg"].values


    plt.figure(figsize=(10, 6))
    plt.hist2d(x, y, bins=50, cmap='Blues')
    plt.colorbar()  # To show the color scale
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.title('2D Histogram of Predicted vs. Target Data')
    plt.show()


