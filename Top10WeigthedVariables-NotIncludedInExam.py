import json
import pandas as pd
import matplotlib.pyplot as plt

# List of file paths
file_paths = [
    'data/mlflow_models/US-CAL-CISO_production_solar/artifacts/feature_importances.json',
    'data/mlflow_models/US-CAL-CISO_production_wind/artifacts/model_params.json',
    'data/mlflow_models/US-TEX-ERCO_production_solar/artifacts/feature_importances.json',
    'data/mlflow_models/US-TEX-ERCO_production_wind/artifacts/model_params.json'
]

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

# Function to load JSON data and plot
def plot_feature_importance_or_model_params(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    df_variables = pd.DataFrame(list(data.items()), columns=['Variable Description', 'Value'])
    df_variables['Value'] = pd.to_numeric(df_variables['Value'], errors='coerce')
    df_variables_sorted = df_variables.sort_values(by='Value', ascending=False)
    df_variables_sorted['Variable Description'] = df_variables_sorted['Variable Description'].apply(abbreviate_name)
    df_variables_top10 = df_variables_sorted.head(10)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(df_variables_top10['Variable Description'], df_variables_top10['Value'], alpha=0.7)
    plt.title(f'Top 10 most weighted variables for {file_path.split("/")[-3]}')
    plt.xlabel('Variable names')
    plt.ylabel('Variable value')
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.25)
    plt.show()

# Apply the function to each file path
for file_path in file_paths:
    plot_feature_importance_or_model_params(file_path)
