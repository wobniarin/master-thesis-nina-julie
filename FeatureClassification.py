import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from the specified path
df = pd.read_parquet('data/features/US-CAL-CISO.parquet')

# Columns in a DataFrame are inherently unique
unique_attribute_names = df.columns.tolist()

# Define a dictionary with group names as keys and criteria as lambda functions as values
criteria_dict = {
    'US-CAL-CISO': lambda col: 'US-CAL-CISO' in col,
    'US-CAL-BANC': lambda col: 'US-CAL-BANC' in col,
    'US-CAL-IID': lambda col: 'US-CAL-IID' in col,
    'US-CAL-LDWP': lambda col: 'US-CAL-LDWP' in col,
    'US-CAL-TIDC': lambda col: 'US-CAL-TIDC' in col,
    'US-NW-BPAT': lambda col: 'US-NW-BPAT' in col,
    'US-NW-PACW': lambda col: 'US-NW-PACW' in col,
    'US-SW-AZPS': lambda col: 'US-SW-AZPS' in col,
    'US-SW-SRP': lambda col: 'US-SW-SRP' in col,
    'US-NW-NEVP': lambda col: 'US-NW-NEVP' in col,
    'US-SW-WALC': lambda col: 'US-SW-WALC' in col,
    'MX': lambda col: col.endswith('MX') and not col.endswith('MX-BC'),
    'MX-BC': lambda col: 'MX-BC' in col,
}

# Dynamically filter columns based on the defined criteria in the dictionary
groups_features = {group: [col for col in df.columns if criterion(col)] for group, criterion in criteria_dict.items()}

# Count the number of features in each group
group_counts = {group: len(features) for group, features in groups_features.items()}

# Sorting groups by their counts (optional)
group_counts = dict(sorted(group_counts.items(), key=lambda item: item[1], reverse=True))

# Creating the bar chart
plt.figure(figsize=(12, 8))
plt.bar(group_counts.keys(), group_counts.values(), color='skyblue')

plt.title('Number of features from zones')
plt.xlabel('Zones')
plt.ylabel('Number of features')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Show the plot
plt.show()


"""US-CAL-CISO
US-CAL-BANC
US-CAL-IID
US-CAL-LDWP
US-CAL-TIDC
US-NW-BPAT
US-NW-PACW
US-SW-AZPS
US-SW-SRP
US-NW-NEVP
US-SW-WALC
interp_MX
MX-BC
"""