import matplotlib.pyplot as plt
import pyarrow.parquet as pq

target_files = [
    'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_target.parquet',
]

zone_capacity_mw = {
    'US-CAL-CISO': {'solar': 19700, 'wind': 6030},
    'US-TEX-ERCO': {'solar': 13500, 'wind': 37000},
}

def split_solar_wind(list_files):
    list = []
    
    for file in list_files:
        zone = file.split('/')[-1].split('_')[0] 
        table = pq.read_table(file)
        df_target = table.to_pandas()
        
        df_target_wind = df_target["power_production_wind_avg"].values / zone_capacity_mw[zone]['wind']
        #df_target_solar = df_target["power_production_solar_avg"].values # with night hours
        df_target_solar = df_target[df_target["power_production_solar_avg"] != 0]["power_production_solar_avg"].values / zone_capacity_mw[zone]['solar'] #with night hours excluded
        
        list.append(df_target_wind)
        list.append(df_target_solar)
    
    return list

# Get wind and solar target values
target_values_list = split_solar_wind(target_files)

# Create a single plot for all boxplots
plt.figure(figsize=(12, 8))

# Create boxplots for each set of target values
for i in range(len(target_values_list)):
    # Calculate position for the boxplot
    position = i * 1.5 + 1
        
    # Create the boxplot
    plt.boxplot(target_values_list[i], positions=[position])

# Set x-axis labels and title
plt.xticks([i * 1.5 + 1 for i in range(len(target_files) * 2)], ['Wind (California)', 'Solar (California)', 'Wind (Texas)', 'Solar (Texas)'])
plt.xlabel('Forecast models')
plt.ylabel('relative value (target values normalized by dividing with installed capacity)')
plt.title('Boxplot of normalized target values for the four forecast models')

# Show the plot
plt.show()