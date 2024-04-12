import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the predicted and target data
predicted_df = pd.read_parquet('data/target_and_predicted/US-CAL-CISO_predicted_nighttime.parquet')
target_df = pd.read_parquet('data/target_and_predicted/US-CAL-CISO_target_nighttime.parquet')

# Ensure the target_time is the index and in the correct datetime format if not already
# This step depends on how your data is structured; adjust as necessary
predicted_df['target_time'] = pd.to_datetime(predicted_df['target_time'], unit='ms')
target_df['target_time'] = pd.to_datetime(target_df['target_time'], unit='ms')
predicted_df.set_index('target_time', inplace=True)
target_df.set_index('target_time', inplace=True)

# Merge the two DataFrames on their index
df = predicted_df.merge(target_df, left_index=True, right_index=True, suffixes=('_pred', '_target'))

# Assuming you have a continuous time series, find the last date in the data
last_date = df.index.max()

# Calculate the first date of the last available week
start_date = last_date - pd.Timedelta(days=6)

# Filter the DataFrame for the last week
last_week_df = df.loc[start_date:last_date]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(last_week_df.index, last_week_df['power_production_solar_avg_pred'], label='Predicted', marker='o')
plt.plot(last_week_df.index, last_week_df['power_production_solar_avg_target'], label='Target', marker='x', linestyle='--')
plt.title('Nighttime Predicted vs. Target Values for Last Available Week')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Load the predicted data from the daytime excluded file
predicted_df = pd.read_parquet('data/target_and_predicted/US-CAL-CISO_predicted_nighttime.parquet')

# Assuming the DataFrame has a 'predicted' column with the values you want to plot
# and a 'target_time' or similar column for the x-axis.

# Convert 'target_time' to a proper datetime format if it's not already
predicted_df['target_time'] = pd.to_datetime(predicted_df['target_time'], unit='ms')

# For a scatter plot, we'll plot the 'target_time' against the 'predicted' values
plt.figure(figsize=(10, 6))
plt.scatter(predicted_df['target_time'], predicted_df['power_production_solar_avg'], s=20, edgecolors= 'b')
plt.title('Scatter Plot of Predicted Values from Nighttime Data California Zone')
plt.xlabel('Time')
plt.ylabel('Predicted Value')
plt.ylim(0, 4100)
plt.xticks(rotation=45)
plt.tight_layout()  # Adjusts subplot params to give some padding
plt.show()
