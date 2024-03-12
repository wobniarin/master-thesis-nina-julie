
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

# Load your time series data
df_combined = pd.read_parquet('data/target_and_predicted/US-CAL-CISO_predicted.parquet')
df_combined['target_time'] = pd.to_datetime(df_combined['target_time'], unit='ms', utc=True)

# Set the index to the target_time
df_combined.set_index('target_time', inplace=True)
df_combined.sort_index(inplace=True)

# Select the data to analyze
data_to_analyze = df_combined['power_production_wind_avg']

# Perform STL decomposition
stl = STL(data_to_analyze, period=24, seasonal=13)
stl_result = stl.fit()

# Plotting each component individually with customizations
plt.figure(figsize=(14, 10))

# Observed
plt.subplot(411)
plt.plot(data_to_analyze)
plt.title('Observed', fontsize=16)
plt.ylim(-1000, 10000) 

# Trend
plt.subplot(412)
plt.plot(stl_result.trend)
plt.title('Trend', fontsize=16)
plt.ylim(0, 6100) 

# Seasonal
plt.subplot(413)
plt.plot(stl_result.seasonal)
plt.title('Seasonality', fontsize=16)
plt.ylim(-600, 800) 

# Residuals
plt.subplot(414)
plt.plot(stl_result.resid, 'o', markersize=2)  # Adjust markersize to make points smaller
plt.title('Residuals', fontsize=16)
plt.ylim(-1500, 1500) 



# Improve spacing between subplots
plt.tight_layout()
plt.show()



