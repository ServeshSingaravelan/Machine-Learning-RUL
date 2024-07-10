import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

file_path = 'C:\Studies\Machine Learning\ML project\Repo\ml-semper-fortis\Datasets\Cleaned Stage 3\B0018_no_outliers.csv'
data = pd.read_csv(file_path)

#here we preprocess the data 
#here, we group the data by the cycle number
capacity_per_cycle = data.groupby('Cycle_Number')['Capacity'].mean()

#since the capacity has noise, we use a smoothing window average to reduce the noise
window_size = 10
smoothed_capacity = capacity_per_cycle.rolling(window=window_size, min_periods=1).mean()

#here we interpolate to fill any missing values in time
interpolated_capacity = smoothed_capacity.interpolate()


starting_capacity = interpolated_capacity.iloc[0]

#here we define criteria for RUL (in this case, 80% of starting capacity)
rul_threshold = 0.8 * starting_capacity

#we find the cycle number where the capacity first drops below the RUL threshold
rul_cycle = interpolated_capacity[interpolated_capacity <= rul_threshold].index[0]

#here we perform feature engineering
#we calculate mean of other relevant features
features_per_cycle = data.groupby('Cycle_Number').mean()

#we calculate two other features that will prove relevant to this case
features_per_cycle['Power'] = features_per_cycle['Voltage_measured'] * features_per_cycle['Current_measured']
features_per_cycle['Energy'] = features_per_cycle['Power'] * features_per_cycle['Time']


features = features_per_cycle[['Voltage_measured', 'Current_measured', 'Energy']]
features['Capacity'] = interpolated_capacity.values
features['Cycle_Number'] = features.index

#here we find rul
features['RUL'] = rul_cycle - features['Cycle_Number']

X = features[['Cycle_Number', 'Capacity', 'Voltage_measured', 'Current_measured', 'Energy']]
y = features['RUL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

model = GradientBoostingRegressor(random_state=12)
model.fit(X_train, y_train)


model_filename = r'C:\Studies\Machine Learning\ML project\Repo\ml-semper-fortis\Regression Models\battery_rul_model_18.joblib'
joblib.dump(model, model_filename)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#we plot original vs smoothed data to get an idea of how appropriate our smoothing is
plt.figure(figsize=(12, 6))
plt.plot(capacity_per_cycle.index, capacity_per_cycle.values, label='Original Capacity', alpha=0.5)
plt.plot(interpolated_capacity.index, interpolated_capacity.values, label='Smoothed & Interpolated Capacity', linewidth=2)
plt.xlabel('Cycle Number')
plt.ylabel('Capacity')
plt.title('Smoothed and Interpolated Battery Capacity Over Cycles')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Predicted vs Actual RUL')
plt.grid(True)
plt.show()

print(f'Starting Capacity: {starting_capacity}')
print(f'RUL Threshold (80% of starting capacity): {rul_threshold}')
print(f'Cycle Number at RUL: {rul_cycle}')

#here is the function for prediction using model
def predict_rul(new_data_path, model_path):

    new_data = pd.read_csv(new_data_path)
    

    new_features_per_cycle = new_data.groupby('Cycle_Number').mean()

    new_features_per_cycle['Power'] = new_features_per_cycle['Voltage_measured'] * new_features_per_cycle['Current_measured']
    new_features_per_cycle['Energy'] = new_features_per_cycle['Power'] * new_features_per_cycle['Time']
    
    new_features_per_cycle = new_features_per_cycle.reset_index()
    
    new_features = new_features_per_cycle[['Cycle_Number', 'Capacity', 'Voltage_measured', 'Current_measured', 'Energy']]
    
    model = joblib.load(model_path)
    
    predicted_rul = model.predict(new_features)
    
    return predicted_rul

new_data_path = 'C:\Studies\Machine Learning\ML project\Repo\ml-semper-fortis\Datasets\Cleaned Stage 3\B0018_no_outliers_test.csv'
model_path = r"C:\Studies\Machine Learning\ML project\Repo\ml-semper-fortis\Regression Models\battery_rul_model_18.joblib"
predicted_rul = predict_rul(new_data_path, model_path)
print(predicted_rul)
