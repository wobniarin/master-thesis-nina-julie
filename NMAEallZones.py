import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import pytz
import numpy as np 
from scipy.stats import spearmanr

# Dictionary mapping predicted file paths to target file paths
target_predicted_files = {
    'data/target_and_predicted/US-CAL-CISO_predicted.parquet': 'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_predicted.parquet': 'data/target_and_predicted/US-TEX-ERCO_target.parquet',
}

naive_CAL = 'naive_forecast_US-CAL-CISO.parquet'
naive_TEX = 'naive_forecast_US-TEX-ERCO.parquet'

# Timezone mapping for each zone
timezone_mapping = {
    'US-CAL-CISO': 'America/Los_Angeles',  # California Time Zone
    'US-TEX-ERCO': 'America/Chicago',      # Texas Time Zone
}

power_types = ['wind', 'solar']

zone_capacity_mw = {
    'US-CAL-CISO': {'solar': 19700, 'wind': 6030},
    'US-TEX-ERCO': {'solar': 13500, 'wind': 37000},
}

def convert_to_local_time(df, zone_key):
    df['target_time'] = pd.to_datetime(df['target_time'], unit='ms', utc=True)
    local_timezone = pytz.timezone(timezone_mapping[zone_key])
    df['target_time'] = df['target_time'].dt.tz_convert(local_timezone)
    return df


def split_horizon(df_predicted, df_target):
    zone_key = df_predicted['zone_key'].iloc[0]
    
    df_predicted = convert_to_local_time(df_predicted, zone_key)
    df_target = convert_to_local_time(df_target, zone_key)
    
    df_predicted = df_predicted[df_predicted["horizon"] == 24]
    df_target = df_target[df_target["horizon"] == 24]
    
    df_combined = pd.merge(df_predicted, df_target, on='target_time', suffixes=('_pred', '_target'))
    
    start_date = pd.Timestamp('2024-01-01', tz=timezone_mapping[zone_key])
    end_date = pd.Timestamp('2024-01-13', tz=timezone_mapping[zone_key])
    
    df_combined = df_combined[(df_combined['target_time'] >= start_date) & (df_combined['target_time'] <= end_date)]
    df_combined.set_index('target_time', inplace=True)
    
    return df_combined


#Calculate daily NMAE normalized by capacity for solar and wind
def nmae(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:
        df_combined[f'abs_error_{power_type}'] = np.abs(df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        
        # If the power type is 'solar', exclude entries where predicted values are zero
        if power_type == 'solar':
            # Filter out rows where the predicted solar production is zero
            df_filtered = df_combined[df_combined[f'power_production_{power_type}_avg_target'] != 0]
            # Calculate NRMSE for solar with non-zero predictions
            df_combined[f'nmae_{power_type}'] = df_filtered[f'abs_error_{power_type}'].resample('D').mean() / zone_capacity_mw[zone][power_type]
        else:
            # Calculate NRMSE normally for other types
            df_combined[f'nmae_{power_type}'] = df_combined[f'abs_error_{power_type}'].resample('D').mean() / zone_capacity_mw[zone][power_type]



        #df_combined[f'abs_error_{power_type}'] = np.abs(df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        #capacity = zone_capacity_mw[zone][power_type]
        #df_combined[f'nmae_{power_type}'] = df_combined[f'abs_error_{power_type}'].resample('D').mean() / capacity
        
    return df_combined


#calculate daily NRMSE normalized by capacity for solar and wind
def nrmse(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:

        # Calculate the errors
        df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])

        """
        # If the power type is 'solar', exclude entries where predicted values are zero
        if power_type == 'solar':
            # Filter out rows where the predicted solar production is zero
            df_filtered = df_combined[df_combined[f'power_production_{power_type}_avg_target'] != 0]
            # Calculate NRMSE for solar with non-zero predictions
            df_combined[f'nrmse_{power_type}'] = np.sqrt((df_filtered[f'error_{power_type}'] ** 2).resample('D').mean()) / zone_capacity_mw[zone][power_type]
        else:
            # Calculate NRMSE normally for other types
            df_combined[f'nrmse_{power_type}'] = np.sqrt((df_combined[f'error_{power_type}'] ** 2).resample('D').mean()) / zone_capacity_mw[zone][power_type]
        """

        df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        capacity = zone_capacity_mw[zone][power_type]
        df_combined[f'nrmse_{power_type}'] = np.sqrt((df_combined[f'error_{power_type}'] ** 2).resample('D').mean()) / capacity
        
    return df_combined


#calculate daily NRMdSE normalized by capacity for solar and wind
def nrmdse(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:
        df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        
        # If the power type is 'solar', exclude entries where predicted values are zero
        if power_type == 'solar':
            # Filter out rows where the predicted solar production is zero
            df_filtered = df_combined[df_combined[f'power_production_{power_type}_avg_target'] != 0]
            # Calculate NRMSE for solar with non-zero predictions
            df_combined[f'nrmdse_{power_type}'] = np.sqrt((df_filtered[f'error_{power_type}'] ** 2).resample('D').median()) / zone_capacity_mw[zone][power_type]
        else:
            # Calculate NRMSE normally for other types
             df_combined[f'nrmdse_{power_type}'] = np.sqrt((df_combined[f'error_{power_type}'] ** 2).resample('D').median()) / zone_capacity_mw[zone][power_type]


        #df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        #capacity = zone_capacity_mw[zone][power_type]
        #df_combined[f'nrmdse_{power_type}'] = np.sqrt((df_combined[f'error_{power_type}'] ** 2).resample('D').median()) / capacity
        
    return df_combined


#calculate daily NMBE normalized by capacity for solar and wind
def nmbe(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:

        df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        
        # If the power type is 'solar', exclude entries where predicted values are zero
        if power_type == 'solar':
            # Filter out rows where the predicted solar production is zero
            df_filtered = df_combined[df_combined[f'power_production_{power_type}_avg_target'] != 0]
            # Calculate NRMSE for solar with non-zero predictions
            df_combined[f'nmbe_{power_type}'] = df_filtered[f'error_{power_type}'].resample('D').mean() / zone_capacity_mw[zone][power_type]
        else:
            # Calculate NRMSE normally for other types
            df_combined[f'nmbe_{power_type}'] = df_combined[f'error_{power_type}'].resample('D').mean() / zone_capacity_mw[zone][power_type]

        
        #df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        #capacity = zone_capacity_mw[zone][power_type]
        #df_combined[f'nmbe_{power_type}'] = df_combined[f'error_{power_type}'].resample('D').mean() / capacity
        
    return df_combined

#calculate daily NMBE normalized by capacity for solar and wind
def nsde(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:

        df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        
        # If the power type is 'solar', exclude entries where predicted values are zero
        if power_type == 'solar':
            # Filter out rows where the predicted solar production is zero
            df_filtered = df_combined[df_combined[f'power_production_{power_type}_avg_target'] != 0]
            # Calculate NRMSE for solar with non-zero predictions
            df_combined[f'nsde_{power_type}'] = df_filtered[f'error_{power_type}'].resample('D').std() / zone_capacity_mw[zone][power_type]
        else:
            # Calculate NRMSE normally for other types
            df_combined[f'nsde_{power_type}'] = df_combined[f'error_{power_type}'].resample('D').std() / zone_capacity_mw[zone][power_type]

        
        #df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        #capacity = zone_capacity_mw[zone][power_type]
        #df_combined[f'nsde_{power_type}'] = df_combined[f'error_{power_type}'].resample('D').std() / capacity
        
    return df_combined

"""
#calculate daily mrae for solar and wind
def mrae(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]
    naive_path = naive_CAL if 'CAL' in zone else naive_TEX
    df_naive = pq.read_table(naive_path).to_pandas()
    df_naive.set_index('target_time', inplace=True)


    for power_type in power_types:
        naive_column = f'naive_forecast_{power_type}'
        forecast_column = f'power_production_{power_type}_avg_pred'
        actual_column = f'power_production_{power_type}_avg_target'

        df_combined = df_combined.join(df_naive[naive_column], how='inner')
        
        # Calculate absolute errors between forecast and actual, and naive and actual
        df_combined['absolute_pred'] = np.abs(df_combined[forecast_column] - df_combined[actual_column])
        df_combined['absolute_naive'] = np.abs(df_combined[naive_column] - df_combined[actual_column])

        #calculate mae for forecasted and naive
        daily_mae_pred = df_combined['absolute_pred'].resample('D').mean()
        daily_mae_naive = df_combined['absolute_naive'].resample('D').mean()

        df_combined[f'mrae_{power_type}'] = daily_mae_pred / daily_mae_naive

    return df_combined
"""

def mrae(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]
    naive_path = naive_CAL if 'CAL' in zone else naive_TEX
    df_naive = pq.read_table(naive_path).to_pandas()
    df_naive['target_time'] = pd.to_datetime(df_naive['target_time'], unit='ms', utc=True)
    df_naive = convert_to_local_time(df_naive, zone)
    df_naive.set_index('target_time', inplace=True)

    for power_type in power_types:
        naive_column = f'naive_forecast_{power_type}'
        forecast_column = f'power_production_{power_type}_avg_pred'
        actual_column = f'power_production_{power_type}_avg_target'

        df_combined = df_combined.join(df_naive[[naive_column]], how='inner')

        if power_type == 'solar':
            filtered_df = df_combined[df_combined[forecast_column] != 0]
            absolute_pred = np.abs(filtered_df[forecast_column] - filtered_df[actual_column])
            absolute_naive = np.abs(filtered_df[naive_column] - filtered_df[actual_column])
        else:
            absolute_pred = np.abs(df_combined[forecast_column] - df_combined[actual_column])
            absolute_naive = np.abs(df_combined[naive_column] - df_combined[actual_column])

        # Calculating daily MAE for forecasted and naive
        daily_mae_pred = absolute_pred.resample('D').mean()
        daily_mae_naive = absolute_naive.resample('D').mean()

        df_combined[f'mrae_{power_type}'] = daily_mae_pred / daily_mae_naive

    return df_combined
    

"""
def spearman(df_combined):
    daily_correlations = {}

    for power_type in power_types:
        # Group by day and compute Spearman's rank correlation for each day
        grouped = df_combined.groupby(pd.Grouper(freq='D'))
        correlations = []
        dates = []

        for name, group in grouped:
            correlation, p_value = spearmanr(group[f'power_production_{power_type}_avg_target'], group[f'power_production_{power_type}_avg_pred'])
            correlations.append(correlation)
            dates.append(name)

        daily_correlations[power_type] = pd.Series(correlations, index=dates)

    # Convert the dictionary to a DataFrame
    df_combined = pd.DataFrame(daily_correlations)
    df_combined.columns = [f'spearman_{col}' for col in df_combined.columns]

    return df_combined
"""

def spearman(df_combined):
    daily_correlations = {}

    for power_type in power_types:
        # Group by day and compute Spearman's rank correlation for each day
        grouped = df_combined.groupby(pd.Grouper(freq='D'))
        correlations = []
        dates = []

        for name, group in grouped:
            if power_type == 'solar':
                # Filter out days where solar predictions are zero
                non_zero_solar = group[group[f'power_production_{power_type}_avg_target'] != 0]
                correlation, p_value = spearmanr(non_zero_solar[f'power_production_{power_type}_avg_target'], non_zero_solar[f'power_production_{power_type}_avg_pred'])
            else:
                correlation, p_value = spearmanr(group[f'power_production_{power_type}_avg_target'], group[f'power_production_{power_type}_avg_pred'])

            correlations.append(correlation)
            dates.append(name)

        daily_correlations[power_type] = pd.Series(correlations, index=dates)

    # Convert the dictionary to a DataFrame and set column names
    df_correlations = pd.DataFrame(daily_correlations)
    df_correlations.columns = [f'spearman_{col}' for col in df_correlations.columns]

    return df_correlations

def metric(predicted_file, target_file, metric_type):
    df_predicted = pq.read_table(predicted_file).to_pandas()
    df_target = pq.read_table(target_file).to_pandas()

    df_combined = split_horizon(df_predicted, df_target)

    if metric_type == 'nmae':  
        return nmae(df_combined)  

    elif metric_type == 'nrmse':  
        return nrmse(df_combined)     

    elif metric_type == 'nrmdse':  
        return nrmdse(df_combined)   
    
    elif metric_type == 'nmbe':  
        return nmbe(df_combined)  
    
    elif metric_type == 'nsde':  
        return nsde(df_combined)
    
    elif metric_type == 'spearman':  
        return spearman(df_combined)
    
    if metric_type == 'mrae':
        return mrae(df_combined)


def visualize_daily_metric_all_zones(metric_type):
    plt.figure(figsize=(12, 6))

    for predicted_file, target_file in target_predicted_files.items():
        zone_df = metric(predicted_file, target_file, metric_type).dropna()
        zone_df.index = zone_df.index.date
                
        zone = 'California' if 'CAL' in predicted_file else 'Texas'

        #plot for wind
        plt.plot(zone_df.index, zone_df[f'{metric_type}_wind'], linestyle='-', marker='o', label=f'Daily {metric_type} for wind in {zone}')

        #plot for solar
        plt.plot(zone_df.index, zone_df[f'{metric_type}_solar'], linestyle='-', marker='o', label=f'Daily {metric_type} for solar in {zone}')


    #Finalize plot
    plt.title(f'Daily {metric_type} for all zones')
    plt.xlabel('Date')
    plt.ylabel(f'{metric_type}')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


#call the function
visualize_daily_metric_all_zones(metric_type='mrae')
