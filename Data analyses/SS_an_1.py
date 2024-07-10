import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
file_paths = [
    'C:\Studies\Machine Learning\ML project\Repo\ml-semper-fortis\Datasets\Cleaned stage 1\B0005_cleaned.csv',
    'C:\Studies\Machine Learning\ML project\Repo\ml-semper-fortis\Datasets\Cleaned stage 2\B0005_cleaned_updated.csv',
    'C:\Studies\Machine Learning\ML project\Repo\ml-semper-fortis\Datasets\Cleaned Stage 3\B0005_no_outliers.csv'
]

datasets = [pd.read_csv(file) for file in file_paths]

# Function to generate scatter plots
def scatter_plots(df, title):
    sns.pairplot(df)
    plt.suptitle(title, y=1.02)
    plt.show()

# Function to generate correlation plots
def correlation_plot(df, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.show()

def calculate_power_energy(df):
    df['Power'] = df['Voltage_measured'] * df['Current_measured']
    df['Energy'] = (df['Power'] * df['Time'].diff().fillna(0)).cumsum()
    return df

# Function to generate statistical summary
def statistical_summary(df, title):
    summary = df.describe()
    print(f"Statistical Summary for {title}:\n", summary)

# Function to generate line plots for "field" vs Time
def line_plots(df, title):
    charge_cycles = df[df['Cycle_type_charge'] == 1]
    discharge_cycles = df[df['Cycle_type_discharge'] == 1]
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=charge_cycles, x='Time', y='Voltage_measured', label='Voltage (Charge)')
    sns.lineplot(data=discharge_cycles, x='Time', y='Voltage_measured', label='Voltage (Discharge)')
    plt.title(f'Voltage vs Time for {title}')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=charge_cycles, x='Time', y='Capacity', label='Capacity (Charge)')
    sns.lineplot(data=discharge_cycles, x='Time', y='Capacity', label='Capacity (Discharge)')
    plt.title(f'Capacity vs Time for {title}')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=charge_cycles, x='Time', y='Current_measured', label='Current (Charge)')
    sns.lineplot(data=discharge_cycles, x='Time', y='Current_measured', label='Current (Discharge)')
    plt.title(f'Current vs Time for {title}')
    plt.legend()
    plt.show()

def capacity_vs_others(df, title):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Capacity', y='Voltage_measured')
    plt.title(f'Capacity vs Voltage_measured for {title}')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Capacity', y='Current_measured')
    plt.title(f'Capacity vs Current_measured for {title}')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Capacity', y='Temperature_measured')
    plt.title(f'Capacity vs Temperature_measured for {title}')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Capacity', y='Power')
    plt.title(f'Capacity vs Power for {title}')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Capacity', y='Energy')
    plt.title(f'Capacity vs Energy for {title}')
    plt.show()



# Analyze each dataset
titles = ["Cleaned Dataset", "Updated Cleaned Dataset", "No Outliers Dataset"]
for df, title in zip(datasets, titles):
    df = calculate_power_energy(df)
    print(f"\nAnalyzing {title}\n")
    statistical_summary(df, title)
    #scatter_plots(df, f"Scatter Plot for {title}")
    correlation_plot(df, f"Correlation Plot for {title}")
    #line_plots(df, title)
    capacity_vs_others(df, title)
