import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import calendar
import numpy as np

target_predicted_files = {
    'US-CAL-CISO': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'US-TEX-ERCO': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

def split_horizon(target_file, horizon):
    df_target = pq.read_table(target_file).to_pandas()
    df_target = df_target[df_target["horizon"] == horizon].copy()
    df_target['target_time'] = pd.to_datetime(df_target['target_time'])
    return df_target

def visualize_seasonality(target_file, horizon, power_type='solar'):
    df_target = split_horizon(target_file, horizon)
    
    # Extract the zone name from the file path
    zone_name = target_file.split('/')[-1].split('_')[0]
    
    df_target['month'] = df_target['target_time'].dt.month
    df_target['year'] = df_target['target_time'].dt.year

    numeric_columns = [f'power_production_{power_type}_avg']
    monthly_avg = df_target.groupby('month')[numeric_columns].mean()

    month_order = list(range(4, 13)) + list(range(1, 4))
    month_names = [calendar.month_name[month] for month in month_order]

    plt.figure(figsize=(12, 6))

    for column_name, color, label in [(f'power_production_{power_type}_avg', 'red', 'Target')]:
        values_to_plot = [monthly_avg.loc[month, column_name] if month in monthly_avg.index else None for month in month_order]
        plt.plot(month_names, values_to_plot, marker='o', linestyle='-', color=color, label=label)

    plt.title(f'Seasonality of {power_type.capitalize()} Power in {zone_name}')
    plt.xlabel('Month')
    plt.ylabel('Average GWh')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylim(1000, 15000)
    plt.tight_layout()
    plt.show()

for zone, target_file in target_predicted_files.items():
    visualize_seasonality(target_file, 12, 'solar')
