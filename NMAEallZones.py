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
    end_date = pd.Timestamp('2024-01-14', tz=timezone_mapping[zone_key])
    
    df_combined = df_combined[(df_combined['target_time'] >= start_date) & (df_combined['target_time'] <= end_date)]
    df_combined.set_index('target_time', inplace=True)
    
    return df_combined


#Calculate daily NMAE normalized by capacity for solar and wind
def nmae(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:
        df_combined[f'abs_error_{power_type}'] = np.abs(df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        capacity = zone_capacity_mw[zone][power_type]
        df_combined[f'nmae_{power_type}'] = df_combined[f'abs_error_{power_type}'].resample('D').mean() / capacity
        
    return df_combined


#calculate daily NRMSE normalized by capacity for solar and wind
def nrmse(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:
        
        df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        capacity = zone_capacity_mw[zone][power_type]
        df_combined[f'nrmse_{power_type}'] = np.sqrt((df_combined[f'error_{power_type}'] ** 2).resample('D').mean()) / capacity
        
    return df_combined


#calculate daily NRMdSE normalized by capacity for solar and wind
def nrmdse(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:
        
        df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        capacity = zone_capacity_mw[zone][power_type]
        df_combined[f'nrmdse_{power_type}'] = np.sqrt((df_combined[f'error_{power_type}'] ** 2).resample('D').median()) / capacity
        
    return df_combined


#calculate daily NMBE normalized by capacity for solar and wind
def nmbe(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:
        
        df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        capacity = zone_capacity_mw[zone][power_type]
        df_combined[f'nmbe_{power_type}'] = df_combined[f'error_{power_type}'].resample('D').mean() / capacity
        
    return df_combined

#calculate daily NMBE normalized by capacity for solar and wind
def nsde(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:
        
        df_combined[f'error_{power_type}'] = (df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target'])
        capacity = zone_capacity_mw[zone][power_type]
        df_combined[f'nsde_{power_type}'] = df_combined[f'error_{power_type}'].resample('D').std() / capacity
        
    return df_combined


#calculate daily Spearman's for solar and wind
def spearman(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]

    for power_type in power_types:
        
        df_combined[f'abs_error_{power_type}'] = (np.abs(df_combined[f'power_production_{power_type}_avg_pred'] - df_combined[f'power_production_{power_type}_avg_target']))
        df_combined[f'spearman_{power_type}'], pvalue = (spearmanr(df_combined[f'power_production_{power_type}_avg_target'], df_combined[f'abs_error_{power_type}'])).resample('D')
        
    return df_combined


def spearman(df_combined):
    zone = df_combined['zone_key_pred'].iloc[0]
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

    # Convert the dictionary to a DataFrame for easy handling later
    df_combined = pd.DataFrame(daily_correlations)
    df_combined.columns = [f'spearman_{col}' for col in df_combined.columns]

    return df_combined



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
    
    elif metric_type == 'mrae':  
        return mrae(df_combined)


def visualize_daily_metric_all_zones(metric_type):
    plt.figure(figsize=(12, 6))

    for predicted_file, target_file in target_predicted_files.items():
        zone_df = metric(predicted_file, target_file, metric_type).dropna()
        
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
    plt.legend()
    plt.tight_layout()
    plt.show()


#call the function
visualize_daily_metric_all_zones(metric_type='spearman')
