import pandas as pd

# Load the previously processed CSV file
file_path = 'E:\Studies\Machine learning\github repo\Machine-Learning-RUL\Datasets\Cleaned stage 2\B0018_cleaned_updated.csv'
df = pd.read_csv(file_path)


# Detect changes in cycle type to identify new cycles
df['Cycle_Change'] = (df['Cycle_type_charge'].diff() != 0) | (df['Cycle_type_discharge'].diff() != 0)
df['Cycle_Number'] = df['Cycle_Change'].cumsum()


cleaned_df = pd.DataFrame()

# Process each cycle independently
for cycle in df['Cycle_Number'].unique():
    cycle_df = df[df['Cycle_Number'] == cycle]
    cleaned_cycle_df = cycle_df
    cleaned_df = pd.concat([cleaned_df, cleaned_cycle_df], ignore_index=True)

# Drop the temporary 'Cycle_Change' column
cleaned_df.drop(columns=['Cycle_Change'], inplace=True)

# Save the cleaned dataframe to a new CSV file
cleaned_df.to_csv('E:\Studies\Machine learning\github repo\Machine-Learning-RUL\Datasets\Cleaned stage 3\B0018_cycle_numbers.csv', index=False)

print("done")

