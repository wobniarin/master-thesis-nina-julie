import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import os
import calendar

target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

def split_horizon(predicted_file, target_file, horizon):
    df_predicted = pq.read_table(predicted_file).to_pandas()
    df_target = pq.read_table(target_file).to_pandas()

    df_predicted = df_predicted[df_predicted["horizon"] == horizon].copy()
    df_target = df_target[df_target["horizon"] == horizon].copy()

    df_predicted['target_time'] = pd.to_datetime(df_predicted['target_time'])
    df_target['target_time'] = pd.to_datetime(df_target['target_time'])

    df_combined = pd.merge(df_predicted, df_target, on='target_time', suffixes=('_pred', '_target'))
    return df_combined

def visualize_seasonality(predicted_file, target_file, horizon, power_type='wind'):
    df_predicted = pq.read_table(predicted_file).to_pandas()
    zone = df_predicted['zone_key'].iloc[0]  # Assuming 'zone_key' is present

    df_combined = split_horizon(predicted_file, target_file, horizon)
    df_combined['error'] = df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target']
    df_combined['abs_error'] = df_combined['error'].abs()
    df_combined['month'] = df_combined['target_time'].dt.month

    # Calculate mean absolute error by month
    error_by_month = df_combined.groupby('month')['abs_error'].mean().reindex(list(range(4,13)) + list(range(1,4)))

    # Create a custom month order (April to January)
    month_order = list(range(4, 13)) + list(range(1, 4))  # From April (4) to December (12), then January (1) to March (3)

    # Reindex error_by_month according to month_order
    error_by_month = error_by_month.reindex(month_order)

    # Plot
    plt.figure(figsize=(10, 6))
    # Convert month numbers back to names for plotting
    month_names = [calendar.month_name[month] for month in month_order]
    plt.plot(month_names, error_by_month, marker='o', linestyle='-', color='blue')
    plt.title(f'Mean Absolute Error by Month for {power_type.capitalize()} Power\nZone: {zone}, Horizon: {horizon}')
    plt.xlabel('Month')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

for predicted_file, target_file in target_predicted_files.items():
    visualize_seasonality(predicted_file, target_file, 12, 'solar')
    visualize_seasonality(predicted_file, target_file, 24, 'solar')
