import matplotlib.pyplot as plt
import pandas as pd


# Load the dataset for 'US-CAL-CISO'
df = pd.read_parquet('naive_forecast_US-CAL-CISO.parquet')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['target_time'], df['power_production_wind_avg'], label='Historic Values')
plt.plot(df['target_time'], df['naive_forecast_wind'], label='Naive Forecast', linestyle='--')

plt.title('Naive Forecast vs Historic Values for US-CAL-CISO')
plt.xlabel('Time')
plt.ylabel('Power Production (Wind, Avg)')
plt.legend()
plt.xticks(rotation=45)  # Rotate dates for better visibility
plt.tight_layout()  # Adjust layout to not cut off labels
plt.show()
