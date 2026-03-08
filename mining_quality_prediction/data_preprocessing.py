import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('Data/mining_data.csv')

# Feature Engineering
df['Iron_to_Silica_Feed_Ratio'] = df['% Iron Feed'] / df['% Silica Feed']
df['Total_Air_Flow'] = df[[col for col in df.columns if 'Flotation Column' in col and 'Air Flow' in col]].sum(axis=1)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = [
    '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow',
    'Ore Pulp pH', 'Ore Pulp Density', 'Iron_to_Silica_Feed_Ratio', 'Total_Air_Flow'
]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split the data into train, test, and validation sets
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
test_data, validation_data = train_test_split(temp_data, test_size=0.33, random_state=42)  # 10% of original data

# Save the splits
train_data.to_csv('Data/mining_train_data.csv', index=False)
test_data.to_csv('Data/mining_test_data.csv', index=False)
validation_data.to_csv('Data/mining_validation_data.csv', index=False)

print("Data preprocessing and feature engineering complete. Train, test, and validation sets have been saved.")
