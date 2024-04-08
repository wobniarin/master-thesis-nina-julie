import plotly.graph_objects as go


import pandas as pd

# Sample data loading
df = pd.read_parquet('data/features/US-CAL-CISO.parquet')

# Search term
search_word = 'export'

# Searching for the term in all text columns and display the first 10 matches
matches = df[df.apply(lambda row: row.astype(str).str.contains(search_word, case=False).any(), axis=1)]

# Check if 'matches' is empty and print accordingly
if matches.empty:
    print("No matches found for the word 'export'.")
else:
    print(matches.head(10))

"""
# Sample data creation
data = {
    'Column1': ['This is an export function', 'No match here', 'Another row', 'exporting data is crucial', 'Just a test', 'Data export completed', 'Nothing here', 'Testing export', 'Random text', 'Export process started', 'Final export'],
    'Column2': ['Random text', 'Some more text', 'Nothing to export here', 'Just another test', 'Data analysis', 'Report generation', 'Finalizing report', 'Preparing data', 'Data export', 'Checking export', 'Export completed']
}
df = pd.DataFrame(data)

# Search for the word "export" (case insensitive) in any text column and display the first 10 matches
search_word = 'export'
matches = df[df.apply(lambda row: row.astype(str).str.contains(search_word, case=False).any(), axis=1)]

# Show the first 10 matches
print(matches.head(10))

# Example zone names
zone_names = [
    'US-CAL-CISO', 'US-CAL-BANC', 'US-CAL-IID', 'US-CAL-LDWP', 'US-CAL-TIDC', 
    'US-NW-BPAT', 'US-NW-PACW', 'US-SW-AZPS', 'US-SW-SRP', 'US-NW-NEVP', 'US-SW-WALC', 'MX', 'MX-BC'
]

# Preparing data for the sunburst plot
ids = zone_names + ['root']  # Adding a 'root' element for the center of the sunburst
parents = ['root' for _ in zone_names] + ['']  # All zones directly under 'root', and 'root' has no parent
labels = zone_names + ['All Zones']  # Labels for each segment

# Create the sunburst chart
fig = go.Figure(go.Sunburst(
    ids=ids,
    parents=parents,
    labels=labels,
))

fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

fig.show()
"""
