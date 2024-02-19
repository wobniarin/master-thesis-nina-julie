import pandas as pd
import matplotlib.pyplot as plt
import json

# Top 10 weighted variable features for US-SW-SRP-wind-production

# File path to the JSON file
file_path_US_SW_SRP_wind_model_params = 'data/mlflow_models/00080cc4f78444059e1962c7f14fd5bc/artifacts/model_params.json'

# Open the JSON file and load it into a Python dictionary
with open(file_path_US_SW_SRP_wind_model_params, 'r') as file:
    model_params = json.load(file)

df_variables = pd.DataFrame(list(model_params.items()), columns=['Variable Description', 'Value'])

# Convert 'Value' to numeric, coercing errors
df_variables['Value'] = pd.to_numeric(df_variables['Value'], errors='coerce')

# Sort the DataFrame based on 'Value' in descending order to get the highest values
df_variables_sorted = df_variables.sort_values(by='Value', ascending=False)

# Abbreviation logic
abbreviations = {
    'forecasted': 'fcast',
    'direction': 'dir',
    'average': 'avg',
    'interpolated': 'interp',
    'lagged': 'lag'
}

def abbreviate_name(name):
    for long_form, short_form in abbreviations.items():
        name = name.replace(long_form, short_form)
    return name

# Apply abbreviation to the 'Variable Description'
df_variables_sorted['Variable Description'] = df_variables_sorted['Variable Description'].apply(abbreviate_name)

# Limit to top 10 entries
df_variables_top10 = df_variables_sorted.head(10)

# Plotting a bar chart for the top 10 entries
plt.figure(figsize=(12, 8))
plt.bar(df_variables_top10['Variable Description'], df_variables_top10['Value'], alpha=0.7)
plt.title('Top 10 most weighted variables for US-SW-SRP Wind')
plt.xlabel('Variable names')
plt.ylabel('Variable value')
plt.xticks(rotation=30, ha="right")  # Adjusted rotation for readability
plt.grid(axis='y', alpha=0.75)
plt.tight_layout(pad=3.0)
plt.subplots_adjust(bottom=0.25)
plt.show()