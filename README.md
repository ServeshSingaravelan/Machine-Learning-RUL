## Prerequisites

python


## Installation and running
```
cd to a folder of your choice
git clone https://mygit.th-deg.de/machine-learning-prok-group/ml-semper-fortis.git

cd to the cloned repo
do: pip install - r requirements.txt

cd to Datasets/predcode
then do python predfinal.py
```
## Methodology Overview
Data Preparation:

After 3 stage preprocessing:

The battery dataset is loaded and grouped by cycle number to calculate the mean for each cycle.
Remaining Useful Life (RUL) is computed as the difference between the maximum cycle number and the current cycle number.
Features for the model are selected by excluding capacity, RUL, and cycle number, and missing values are filled with zeros.

Model Selection:

A Random Forest Regressor is chosen for its robustness and ability to handle complex relationships in the data.
The model is initialized with a fixed random seed for reproducibility.
Feature Selection:

Sequential Forward Selection (SFS) is used with a time-series split cross-validation to select the most relevant features. SFS iteratively adds features that improve model performance until the optimal set is found.

Hyperparameter Tuning:

GridSearchCV is employed with a predefined parameter grid to find the best hyperparameters for the Random Forest model, using time-series cross-validation to maintain the temporal order of data.

Model Evaluation:

The data is split into training and testing sets using the last split from the time-series split.
The best model is evaluated on both training and testing sets using R-squared scores and Root Mean Squared Error (RMSE).
Feature importance is analyzed and visualized to understand the contribution of each selected feature.

Model Persistence:

The best Random Forest model and the list of selected features are saved for future predictions.

Prediction Function:

A prediction function is defined to load a new dataset, preprocess it, and make predictions using the saved model. It also evaluates and visualizes the performance of the model on the new data.

Key Points
Model Used: Random Forest Regressor
Reason: Robustness and ability to capture complex interactions in the data.
Feature Selection: Sequential Forward Selection (SFS) with TimeSeriesSplit
Reason: To select the most relevant features while considering the temporal structure of the data.
Hyperparameter Tuning: GridSearchCV with TimeSeriesSplit
Reason: To find the optimal hyperparameters for the Random Forest model.
Evaluation Metrics: R-squared and RMSE
Reason: To assess the model's performance in predicting the RUL accurately.
Visualization: Feature importance and Actual vs Predicted plots
Reason: To interpret the model's decisions and assess its accuracy visually.
