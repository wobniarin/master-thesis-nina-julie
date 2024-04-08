import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from ExcludeNighttime import exclude_nighttimes

target_files = [
    'data/target_and_predicted/US-CAL-CISO_target.parquet',
    'data/target_and_predicted/US-TEX-ERCO_target.parquet',
]

def split_solar_wind(list_files):
    list = []
    
    for file in list_files:
        table = pq.read_table(file)
        df_target = table.to_pandas()
        

        df_target_wind = df_target["power_production_wind_avg"].values
        #df_target_solar = df_target["power_production_solar_avg"].values
        df_target_solar = df_target[df_target["power_production_solar_avg"] != 0]["power_production_solar_avg"].values 
        
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
plt.xticks([i * 1.5 + 1 for i in range(len(target_files) * 2)], ['Wind (US-CAL-CISO)', 'Solar (US-CAL-CISO)', 'Wind (US-TEX-ERCO)', 'Solar (US-TEX-ERCO)'])
plt.xlabel('Target Values')
plt.ylabel('Values')
plt.title('Boxplot of Target Values')

# Show the plot
plt.show()