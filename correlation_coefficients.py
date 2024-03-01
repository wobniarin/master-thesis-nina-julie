from scipy.stats import pearsonr, spearmanr
from Scatterplot import split_horizon

target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

for predicted_file, target_file in target_predicted_files.items():
    df_predicted_12, df_target_12 = split_horizon(predicted_file, target_file, 12)

    # Set "target_time" as index for both DataFrames
    df_predicted_12.set_index(['target_time'], inplace=True)
    df_target_12.set_index(['target_time'], inplace=True)

    df_predicted_solar = df_predicted_12.dropna(subset=["power_production_solar_avg"])
    df_target_solar = df_target_12.dropna(subset=["power_production_solar_avg"])

    # Reindex or align one DataFrame to match the other
    df_target_solar = df_target_solar.reindex(df_predicted_solar.index)

    x = df_predicted_solar["power_production_solar_avg"].values
    y = df_target_solar["power_production_solar_avg"].values

    # Calculate Pearson's correlation
    corr, p_value = pearsonr(x, y)

    print(f"Pearson's correlation coefficient: {corr}")
    print(f"P-value: {p_value}")

    corr, p_value = spearmanr(x, y)

    print(f"Spearman's rank correlation coefficient: {corr}")
    print(f"P-value: {p_value}")

