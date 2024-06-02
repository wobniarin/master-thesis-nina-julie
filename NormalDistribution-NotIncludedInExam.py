import pyarrow.parquet as pq
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import skew, kurtosis

target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

    
def predicted_horizon_split():
    predicted_list = []

    for file in target_predicted_files:
        table = pq.read_table(file)
        df_predicted = table.to_pandas()
        df_predicted_12 = df_predicted[df_predicted["horizon"] == 12]
        df_predicted_24 = df_predicted[df_predicted["horizon"] == 24]
        predicted_list.append(df_predicted_12)
        predicted_list.append(df_predicted_24)
    
    return predicted_list


def split_solar_wind(list_files):
    solar_wind_list = []
    
    for predicted, target in list_files.items():
        predicted = pq.read_table(predicted).to_pandas()
        target = pq.read_table(target).to_pandas()

        # Set "target_time" as index for both DataFrames
        predicted.set_index(['target_time'], inplace=True)
        target.set_index(['target_time'], inplace=True)

        filtered_target = target[target['power_production_solar_avg'] != 0]
        filtered_predicted = predicted.loc[filtered_target.index]    

        # Drop nan values only for specified columns
        cols_to_drop_nan = ["power_production_solar_avg", "power_production_wind_avg"]
        for col in cols_to_drop_nan:
            filtered_predicted = filtered_predicted.dropna(subset=[col])    

        solar_wind_dict = {
            'predicted_solar': filtered_predicted["power_production_solar_avg"].values,
            'target_solar': filtered_target["power_production_solar_avg"].values,
            'predicted_wind': filtered_predicted["power_production_wind_avg"].values,
            'target_wind': filtered_target["power_production_wind_avg"].values
        }
        solar_wind_list.append(solar_wind_dict)

    return solar_wind_list


def plot_normal_distributions(data_list):
    # Determine common axis limits

    for data_dict in data_list:
        # Plot for solar
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        sns.kdeplot(data_dict['predicted_solar'], label='Predicted Solar', color='red')
        sns.kdeplot(data_dict['target_solar'], label='Target Solar', color='blue')
        plt.title(f'Solar Energy Distribution')
        plt.xlabel('Power Production')
        plt.ylabel('Density')
        plt.legend()
        print("Skewness - Predicted Solar:", skew(data_dict['predicted_solar']))
        print("Kurtosis - Predicted Solar:", kurtosis(data_dict['predicted_solar']))
        print("Skewness - Target Solar:", skew(data_dict['target_solar']))
        print("Kurtosis - Target Solar:", kurtosis(data_dict['target_solar']))

        # Plot for wind
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        sns.kdeplot(data_dict['predicted_wind'], label='Predicted Wind', color='orange')
        sns.kdeplot(data_dict['target_wind'], label='Target Wind', color='green')
        plt.title(f'Wind Energy Distribution')
        plt.xlabel('Power Production')
        plt.ylabel('Density')
        plt.legend()
        print("Skewness - Predicted Wind:", skew(data_dict['predicted_wind']))
        print("Kurtosis - Predicted Wind:", kurtosis(data_dict['predicted_wind']))
        print("Skewness - Target Wind:", skew(data_dict['target_wind']))
        print("Kurtosis - Target Wind:", kurtosis(data_dict['target_wind']))

        plt.tight_layout()
        plt.show()

plot_normal_distributions(split_solar_wind(target_predicted_files))


