import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'E:\Studies\Machine learning\github repo\Machine-Learning-RUL\Datasets\Cleaned Stage 3\B0005_cycle_numbers.csv'
df = pd.read_csv(file_path)

# Columns to plot
columns_to_plot = [
    'Voltage_measured', 'Current_measured', 'Capacity', 
    'Current_charge', 'Voltage_charge', 'Current_load', 'Voltage_load'
]

cycle_number = df['Cycle_Number']

# Plotting the data
plt.figure(figsize=(14, 10))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(4, 2, i)
    plt.plot(cycle_number, df[column], label=column)
    plt.xlabel('Cycle Number')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
