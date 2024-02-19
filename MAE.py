import pandas as pd
import pyarrow.parquet as pq

file_path_predicted = 'data/target_and_predicted/US-CAL-BANC_predicted.parquet'
file_path_predicted = 'data/target_and_predicted/US-CAL-BANC_predicted.parquet'
table = pq.read_table(file_path_predicted)
df_predicted = table.to_pandas()
print(df_predicted.head)


file_path_target = 'data/target_and_predicted/US-CAL-BANC_target.parquet'
table = pq.read_table(file_path_target)
df_target = table.to_pandas()
print(df_target.head)

mae = df_predicted["power_production_solar_avg"].sub(df_target["power_production_solar_avg"]).abs().mean()

print("Mean Absolute Error (MAE):", mae)



