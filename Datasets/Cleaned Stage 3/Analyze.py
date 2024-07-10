import pandas as pd

# Load the previously processed CSV file
file_path = 'D:\chrome downloads\B0018\Datasets cleaned converted\B0018_cleaned_updated.csv'
df = pd.read_csv(file_path)


# Detect changes in cycle type to identify new cycles
df['Cycle_Change'] = (df['Cycle_type_charge'].diff() != 0) | (df['Cycle_type_discharge'].diff() != 0)
df['Cycle_Number'] = df['Cycle_Change'].cumsum()

# Function to detect outliers using IQR and replace them with median values
def replace_outliers_with_median(cycle_df):
    numeric_cols = cycle_df.select_dtypes(include=['number']).columns
    Q1 = cycle_df[numeric_cols].quantile(0.25)
    Q3 = cycle_df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 6 * IQR  # Adjusted lower bound to be 3 times IQR
    upper_bound = Q3 + 6 * IQR  # Adjusted upper bound to be 3 times IQR
    is_outlier = (cycle_df[numeric_cols] < lower_bound) | (cycle_df[numeric_cols] > upper_bound)
    
    for col in numeric_cols:
        if is_outlier[col].any():
            median_value = cycle_df.loc[~is_outlier[col], col].median()
            cycle_df.loc[is_outlier[col], col] = median_value
            
    return cycle_df


cleaned_df = pd.DataFrame()

# Process each cycle independently
for cycle in df['Cycle_Number'].unique():
    cycle_df = df[df['Cycle_Number'] == cycle]
    cleaned_cycle_df = replace_outliers_with_median(cycle_df)
    cleaned_df = pd.concat([cleaned_df, cleaned_cycle_df], ignore_index=True)

# Drop the temporary 'Cycle_Change' column
cleaned_df.drop(columns=['Cycle_Change'], inplace=True)

# Save the cleaned dataframe to a new CSV file
cleaned_df.to_csv('D:\chrome downloads\B0018\Datasets cleaned converted\B0018_no_outliers.csv', index=False)

print("done")

