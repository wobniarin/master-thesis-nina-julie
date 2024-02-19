import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

def plot(x_value, y_value):
    max_y = max(y_value)
    max_x = max(x_value)
    maximum_y_or_x = max(max_y, max_x)
    plt.plot([0, maximum_y_or_x], [0, maximum_y_or_x], color='red')
    plt.scatter(x_value, y_value, s=10)
    plt.ylabel('Target')
    plt.xlabel('Predicted')
    print(plt.show())

for predicted_file, target_file in target_predicted_files.items():
    table = pq.read_table(predicted_file)
    df_predicted = table.to_pandas()
    df_predicted = pd.DataFrame(df_predicted)
    df_predicted = df_predicted.dropna()

    table = pq.read_table(target_file)
    df_target = table.to_pandas()
    df_target = pd.DataFrame(df_target)
    df_target = df_target.dropna()

    df_target = df_target.sample(n=len(df_predicted), random_state=42)

    #wind
    x = df_predicted["power_production_wind_avg"].values
    y = df_target["power_production_wind_avg"].values
    plot(x,y)

    #solar
    x = df_predicted["power_production_solar_avg"].values
    y = df_target["power_production_solar_avg"].values
    plot(x,y)

