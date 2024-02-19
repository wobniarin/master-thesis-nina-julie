import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

file_path_predicted = 'data/target_and_predicted/US-CAL-BANC_predicted.parquet'
table = pq.read_table(file_path_predicted)
df_predicted = table.to_pandas()
df_predicted = pd.DataFrame(df_predicted)
df_predicted = df_predicted.dropna()


file_path_target = 'data/target_and_predicted/US-CAL-BANC_target.parquet'
table = pq.read_table(file_path_target)
df_target = table.to_pandas()
df_target = pd.DataFrame(df_target)
df_target = df_target.dropna()

df_target = df_target.sample(n=len(df_predicted), random_state=42)

x = df_predicted["power_production_solar_avg"].values
x_reshaped = x.reshape(-1,1)
y = df_target["power_production_solar_avg"].values


# Fit the linear regression model
model = LinearRegression().fit(x_reshaped, y)

# Get the slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

plt.scatter(x, y, color='blue', label='Predicted')
plt.scatter(x, y, color='red', label='Target')
plt.plot(x, [slope*val + intercept for val in x], color='black', label='Linear Regression')
plt.ylabel('Target')
plt.xlabel('Predicted')
plt.legend()  # Adding a legend to distinguish between target and predicted
print(plt.show())




"""
#create basic scatterplot
plt.plot(x, y)

#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(x, y, 1)

#add linear regression line to scatterplot 
print(plt.plot(x, m*x+b))
"""
