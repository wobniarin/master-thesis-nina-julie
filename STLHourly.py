import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.dates as mdates

# Load your time series data
df_combined = pd.read_parquet('data/target_and_predicted/US-CAL-CISO_predicted.parquet')
df_combined['target_time'] = pd.to_datetime(df_combined['target_time'], unit='ms', utc=True)

# Set the index to the target_time
df_combined.set_index('target_time', inplace=True)
df_combined.sort_index(inplace=True)

# Check for NaN values in your series
print("NaN values before filling:", df_combined['power_production_solar_avg'].isnull().sum())

# Option to fill NaN values if necessary
# Uncomment the line below to fill NaN values with the previous value (forward fill)
df_combined['power_production_solar_avg'].fillna(method='ffill', inplace=True)

# Select the data to analyze
data_to_analyze = df_combined['power_production_solar_avg']

# Perform STL decomposition
stl = STL(data_to_analyze, period=24, seasonal=13)
stl_result = stl.fit()

# Plotting each component individually with customizations
plt.figure(figsize=(14, 10))

# Configure date formatting
months = mdates.MonthLocator()  # every month
month_fmt = mdates.DateFormatter('%Y-%m')

# Function to set up each subplot
def setup_subplot(ax, data, title, y_limit=None):
    ax.plot(data)
    ax.set_title(title, fontsize=16)
    if y_limit:
        ax.set_ylim(y_limit)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(month_fmt)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Trend
ax1 = plt.subplot(411)
setup_subplot(ax1, stl_result.trend, 'Trend', (0, 6100))

# Seasonal
ax2 = plt.subplot(412)
setup_subplot(ax2, stl_result.seasonal, 'Seasonality', (-600, 800))

# Residuals
ax3 = plt.subplot(413)
setup_subplot(ax3, stl_result.resid, 'Residuals', (-1500, 1500))

# Observed
ax4 = plt.subplot(414)
setup_subplot(ax4, data_to_analyze, 'Observed', (-200, 6200))

# Improve spacing between subplots
plt.tight_layout()
plt.show()
