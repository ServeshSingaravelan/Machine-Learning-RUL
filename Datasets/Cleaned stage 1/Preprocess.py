import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#In this script, we remove impedence cycle and related fields, timestamp, and we encode cycles to prepare for ML


file_path = 'B0005_converted.csv'
data = pd.read_csv(file_path)

# Remove rows where the cycle type is 'impedance'
data = data[data['Cycle_type'] != 'impedance']

# Drop the "timestamp" column
if 'Timestamp' in data.columns:
    data.drop(columns=['Timestamp'], inplace=True)

# Remove columns related to impedance measurements
impedance_related_columns = ['Sense_current', 'Battery_current', 'Current_ratio', 'Battery_impedance', 'Rectified_Impedance', 'Re', 'Rct']
data.drop(columns=impedance_related_columns, inplace=True, errors='ignore')


# Preparing for one-hot encoding of the categorical 'Cycle_type' column
encoder = OneHotEncoder()

# Creating a column transformer to apply one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, ['Cycle_type'])
    ],
    remainder='passthrough'  # Keep all other columns unchanged
)

# Applying transformations
data_transformed = preprocessor.fit_transform(data)

# Getting the new feature names from the encoder, and keeping the rest of the column names unchanged
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['Cycle_type'])
all_feature_names = list(encoded_feature_names) + [col for col in data.columns if col != 'Cycle_type']

# Creating a DataFrame from the transformed data
transformed_data_df = pd.DataFrame(data_transformed, columns=all_feature_names)

output_file_path = 'B0005_cleaned.csv'
transformed_data_df.to_csv(output_file_path, index=False)

print("Data saved to:", output_file_path)
