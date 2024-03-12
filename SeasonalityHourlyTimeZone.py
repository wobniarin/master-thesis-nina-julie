import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load your time series data
df_combined = pd.read_parquet('data/target_and_predicted/US-CAL-CISO_target.parquet')
df_combined['target_time'] = pd.to_datetime(df_combined['target_time'], unit='ms', utc=True)
df_combined.set_index('target_time', inplace=True)
df_combined.sort_index(inplace=True)

# Resample data to a weekly frequency, taking the mean for each week
df_weekly = df_combined['power_production_wind_avg'].resample('W').mean()

# Perform time series decomposition with a weekly cycle
result = seasonal_decompose(df_weekly, model='additive', period=7)

# Plot the decomposed components of the time series
plt.figure(figsize=(14, 10))

# Plot the trend component
plt.subplot(411)
plt.plot(result.trend)
plt.title('Trend')

# Plot the seasonal component
plt.subplot(412)
plt.plot(result.seasonal)
plt.title('Seasonality')
plt.ylim(-300, 600)  # Set the y-axis limits for seasonality plot

# Plot the residual component
plt.subplot(413)
plt.plot(result.resid)
plt.title('Residuals')

# Plot the observed data
plt.subplot(414)
plt.plot(df_combined['power_production_wind_avg'])
plt.title('Observed')

plt.tight_layout()
plt.show()