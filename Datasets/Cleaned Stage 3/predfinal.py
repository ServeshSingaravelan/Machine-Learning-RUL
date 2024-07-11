import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import numpy as np

file_path = r"E:\Studies\Machine learning\github repo\Machine-Learning-RUL\Datasets\Cleaned Stage 3\B0005_cycle_numbers.csv"
battery_data = pd.read_csv(file_path)

#here we group by cyc number
cycle_data = battery_data.groupby('Cycle_Number').mean().reset_index()

#here we calculate Remaining Useful Life (RUL)
cycle_data['RUL'] = cycle_data['Cycle_Number'].max() - cycle_data['Cycle_Number']

#here we select features and target
features = cycle_data.drop(columns=['Capacity', 'RUL', 'Cycle_Number'])
target = cycle_data['RUL']


features_filled = features.fillna(0)

rf = RandomForestRegressor(random_state=42)

#here we do forward feature selection with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=4)
sfs = SequentialFeatureSelector(rf, n_features_to_select='auto', direction='forward', cv=tscv, n_jobs=-1)

#here we fit the feature selector
sfs.fit(features_filled, target)


selected_features = features_filled.columns[sfs.get_support()]

#param grid hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [5, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4,8],
}

#we perform GridSearchCV with TimeSeriesSplit
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(features_filled[selected_features], target)


best_rf = grid_search.best_estimator_

#here we split the data into training and testing sets using the last split for testing
train_index, test_index = list(tscv.split(features_filled))[-1]
X_train, X_test = features_filled.iloc[train_index], features_filled.iloc[test_index]
y_train, y_test = target.iloc[train_index], target.iloc[test_index]

# we evaluate the best model
train_score = best_rf.score(X_train[selected_features], y_train)
test_score = best_rf.score(X_test[selected_features], y_test)
train_rmse = mean_squared_error(y_train, best_rf.predict(X_train[selected_features]), squared=False)
test_rmse = mean_squared_error(y_test, best_rf.predict(X_test[selected_features]), squared=False)

#here we display the best parameters and evaluation metrics
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)
print("Training score: ", train_score)
print("Testing score: ", test_score)
print("Training RMSE: ", train_rmse)
print("Testing RMSE: ", test_rmse)

#here we do feature Importance Analysis
feature_importances = best_rf.feature_importances_
importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

#we plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()

#here we plot Predicted vs Actual
y_pred_train = best_rf.predict(X_train[selected_features])
y_pred_test = best_rf.predict(X_test[selected_features])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Data: Actual vs Predicted')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Data: Actual vs Predicted')

plt.tight_layout()
plt.show()

#here we plot Predicted vs Actual Capacity as a line graph
plt.figure(figsize=(12, 6))

#plots for training data
plt.subplot(1, 2, 1)
plt.plot(X_train.index, y_train, label='Actual')
plt.plot(X_train.index, y_pred_train, label='Predicted')
plt.xlabel('Cycle Number')
plt.ylabel('Remaining Useful Life (RUL)')
plt.title('Training Data: Actual vs Predicted Capacity')
plt.legend()

#plots for testing data
plt.subplot(1, 2, 2)
plt.plot(X_test.index, y_test, label='Actual')
plt.plot(X_test.index, y_pred_test, label='Predicted')
plt.xlabel('Cycle Number')
plt.ylabel('Remaining Useful Life (RUL)')
plt.title('Test Data: Actual vs Predicted Capacity')
plt.legend()

plt.tight_layout()
plt.show()

"""
this stuff, you can ignore, I just have it here in case, but we prolly wont be using it, so please ignore
# Further Hyperparameter Tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, cv=tscv, n_iter=100, n_jobs=-1, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(features_filled[selected_features], target)

#we get the best model from random search
best_rf_random = random_search.best_estimator_

#we evaluate the best model from random search
train_score_random = best_rf_random.score(X_train[selected_features], y_train)
test_score_random = best_rf_random.score(X_test[selected_features], y_test)
train_rmse_random = mean_squared_error(y_train, best_rf_random.predict(X_train[selected_features]), squared=False)
test_rmse_random = mean_squared_error(y_test, best_rf_random.predict(X_test[selected_features]), squared=False)

#we display the best parameters and evaluation metrics
best_params_random = random_search.best_params_
print("After a bit more hyperparameter tuning, Best parameters found: ", best_params_random)
print("Training score (Random Search): ", train_score_random)
print("Testing score (Random Search): ", test_score_random)
print("Training RMSE (Random Search): ", train_rmse_random)
print("Testing RMSE (Random Search): ", test_rmse_random)

#plot Predicted vs Actual for the model from RandomizedSearchCV
y_pred_train_random = best_rf_random.predict(X_train[selected_features])
y_pred_test_random = best_rf_random.predict(X_test[selected_features])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train_random, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Data: Actual vs Predicted (Random Search)')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test_random, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Data: Actual vs Predicted (Random Search)')

plt.tight_layout()
plt.show()

#we plot Predicted vs Actual Capacity as a Line Graph for RandomizedSearchCV Model
plt.figure(figsize=(12, 6))

#training data
plt.subplot(1, 2, 1)
plt.plot(X_train.index, y_train, label='Actual')
plt.plot(X_train.index, y_pred_train_random, label='Predicted')
plt.xlabel('Cycle Number')
plt.ylabel('Remaining Useful Life (RUL)')
plt.title('Training Data: Actual vs Predicted Capacity (Random Search)')
plt.legend()

#testing data
plt.subplot(1, 2, 2)
plt.plot(X_test.index, y_test, label='Actual')
plt.plot(X_test.index, y_pred_test_random, label='Predicted')
plt.xlabel('Cycle Number')
plt.ylabel('Remaining Useful Life (RUL)')
plt.title('Test Data: Actual vs Predicted Capacity (Random Search)')
plt.legend()

plt.tight_layout()
plt.show() 

end of "ignore"

"""

joblib.dump(best_rf, r"E:\Studies\Machine learning\github repo\Machine-Learning-RUL\Regression Models\battery_5_best_rf_model.pkl")

#we save the selected features to use when calling the model for prediction
selected_features_list = list(selected_features)
joblib.dump(selected_features_list, r"E:\Studies\Machine learning\github repo\Machine-Learning-RUL\Regression Models\battery_5_selected_features.pkl")



def predictor(file_path, model_path, features_path, num_rows):
    #load the dataset
    new_data = pd.read_csv(file_path)
    
    #we prepare the data by aggregating per cycle
    cycle_data = new_data.groupby('Cycle_Number').mean().reset_index()

    #calculate Remaining Useful Life (RUL)
    cycle_data['RUL'] = cycle_data['Cycle_Number'].max() - cycle_data['Cycle_Number']

    #select features and target
    features = cycle_data.drop(columns=['Capacity', 'RUL', 'Cycle_Number'])
    target = cycle_data['RUL']

    #replace NaNs with zeros
    features_filled = features.fillna(0)
    
    #load the selected features
    selected_features = joblib.load(features_path)
    
    #take the first num_rows for prediction
    X_new = features_filled[selected_features].iloc[:num_rows]
    y_actual = target.iloc[:num_rows]

    #we load the saved model
    model = joblib.load(model_path)

    #we make predictions
    y_pred = model.predict(X_new)

    #we evaluate the model
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    #we display the RMSE
    print("RMSE for the new dataset:", rmse)

    #we plot Predicted vs Actual
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_actual, y_pred, alpha=0.5)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')

    plt.subplot(1, 2, 2)
    plt.plot(y_actual.index, y_actual, label='Actual')
    plt.plot(y_actual.index, y_pred, label='Predicted')
    plt.xlabel('Cycle Number')
    plt.ylabel('Remaining Useful Life (RUL)')
    plt.title('Actual vs Predicted Capacity')
    plt.legend()

    plt.tight_layout()
    plt.show()

#I have a reference call here, to show the format 
predictor(r"E:\Studies\Machine learning\github repo\Machine-Learning-RUL\Datasets\Cleaned stage 3\B0005_cycle_numbers.csv", r"E:\Studies\Machine learning\github repo\Machine-Learning-RUL\Regression Models\battery_5_best_rf_model.pkl", r"E:\Studies\Machine learning\github repo\Machine-Learning-RUL\Regression Models\battery_5_selected_features.pkl", 50000) 