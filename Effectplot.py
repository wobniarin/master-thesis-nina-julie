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

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)

    df_variables = pd.DataFrame(list(data.items()), columns=['Variable Description', 'Value'])
    df_variables['Value'] = pd.to_numeric(df_variables['Value'], errors='coerce')

    print(df_variables['Value'])

    