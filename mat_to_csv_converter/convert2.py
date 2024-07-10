import scipy.io
import pandas as pd
import numpy as np

# load file
file_path = 'FCB0018.mat'
mat = scipy.io.loadmat(file_path)

# main mat ext
data = mat['B0018']

# ext cycle types
data_element = data[0, 0]
cycle_data = data_element['cycle']

# list to store cycles
all_cycles = []

# each cycle entry iterate
for i in range(cycle_data.shape[1]):
    cycle_entry = cycle_data[0, i]
    cycle_type = cycle_entry['type'][0]
    ambient_temperature = cycle_entry['ambient_temperature'][0][0]
    time_data = cycle_entry['time'][0]
    data_data = cycle_entry['data'][0][0]

    # df for each cycle data
    if cycle_type == 'discharge':
        max_length = max(len(data_data['Voltage_measured'].flatten()),
                         len(data_data['Current_measured'].flatten()),
                         len(data_data['Temperature_measured'].flatten()),
                         len(data_data['Current_load'].flatten()),
                         len(data_data['Voltage_load'].flatten()),
                         len(data_data['Capacity'].flatten()),
                         len(data_data['Time'].flatten()))
        
        cycle_df = pd.DataFrame({
            'Voltage_measured': np.pad(data_data['Voltage_measured'].flatten(), (0, max_length - len(data_data['Voltage_measured'].flatten())), constant_values=np.nan),
            'Current_measured': np.pad(data_data['Current_measured'].flatten(), (0, max_length - len(data_data['Current_measured'].flatten())), constant_values=np.nan),
            'Temperature_measured': np.pad(data_data['Temperature_measured'].flatten(), (0, max_length - len(data_data['Temperature_measured'].flatten())), constant_values=np.nan),
            'Current_load': np.pad(data_data['Current_load'].flatten(), (0, max_length - len(data_data['Current_load'].flatten())), constant_values=np.nan),
            'Voltage_load': np.pad(data_data['Voltage_load'].flatten(), (0, max_length - len(data_data['Voltage_load'].flatten())), constant_values=np.nan),
            'Capacity': np.pad(data_data['Capacity'].flatten(), (0, max_length - len(data_data['Capacity'].flatten())), constant_values=np.nan),  # adding padding for Capacity field
            'Time': np.pad(data_data['Time'].flatten(), (0, max_length - len(data_data['Time'].flatten())), constant_values=np.nan)
        })
    elif cycle_type == 'charge':
        cycle_df = pd.DataFrame({
            'Voltage_measured': data_data['Voltage_measured'].flatten(),
            'Current_measured': data_data['Current_measured'].flatten(),
            'Temperature_measured': data_data['Temperature_measured'].flatten(),
            'Current_charge': data_data['Current_charge'].flatten(),
            'Voltage_charge': data_data['Voltage_charge'].flatten(),
            'Time': data_data['Time'].flatten()
        })
    elif cycle_type == 'impedance':
        # max length of all fields
        max_length = max(len(data_data['Sense_current'].flatten()),
                         len(data_data['Battery_current'].flatten()),
                         len(data_data['Current_ratio'].flatten()),
                         len(data_data['Battery_impedance'].flatten()),
                         len(data_data['Rectified_Impedance'].flatten()))

        # pad missing with NaN
        cycle_df = pd.DataFrame({
            'Sense_current': np.pad(data_data['Sense_current'].flatten(), (0, max_length - len(data_data['Sense_current'].flatten())), constant_values=np.nan),
            'Battery_current': np.pad(data_data['Battery_current'].flatten(), (0, max_length - len(data_data['Battery_current'].flatten())), constant_values=np.nan),
            'Current_ratio': np.pad(data_data['Current_ratio'].flatten(), (0, max_length - len(data_data['Current_ratio'].flatten())), constant_values=np.nan),
            'Battery_impedance': np.pad(data_data['Battery_impedance'].flatten(), (0, max_length - len(data_data['Battery_impedance'].flatten())), constant_values=np.nan),
            'Rectified_Impedance': np.pad(data_data['Rectified_Impedance'].flatten(), (0, max_length - len(data_data['Rectified_Impedance'].flatten())), constant_values=np.nan),
            'Re': [data_data['Re'][0, 0]] * max_length,
            'Rct': [data_data['Rct'][0, 0]] * max_length
        })
    else:
        continue

    # add cycle type and temp to df
    cycle_df['Cycle_type'] = cycle_type
    cycle_df['Ambient_temperature'] = ambient_temperature

    # time date [year, month, day, hour, minute, second]
    time_structured = time_data.reshape(-1, 6)
    cycle_df['Timestamp'] = pd.to_datetime({
        'year': time_structured[:, 0],
        'month': time_structured[:, 1],
        'day': time_structured[:, 2],
        'hour': time_structured[:, 3],
        'minute': time_structured[:, 4],
        'second': time_structured[:, 5]
    })

    # add to cycle list
    all_cycles.append(cycle_df)

# concat all cyc df
all_cycles_df = pd.concat(all_cycles, ignore_index=True)

csv_file_path = 'B0018_converted.csv'
all_cycles_df.to_csv(csv_file_path, index=False)

print(f"CSV file has been saved to: {csv_file_path}")
