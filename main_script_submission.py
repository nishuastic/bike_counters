# %%
import subprocess

subprocess.run(
    [
        "pip",
        "install",
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "holidays",
    ]
)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import holidays
# import seaborn as sns

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

from datetime import datetime

# %%
import utils

X, y = utils.get_train_data()
X.head(2)


# %%
def _encode_dates(X):
    X = X.copy()  # Modify a copy of X

    # Ensure 'date' is in datetime format
    X["date"] = pd.to_datetime(X["date"])

    # Extract date components
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    # Identify weekends (Saturday = 5, Sunday = 6)
    X["is_weekend"] = X["weekday"].isin([5, 6])

    # Get French holidays for all years in the dataset
    years = X["year"].unique()
    fr_holidays = holidays.France(years=years)

    # Identify holidays
    X["is_holiday"] = X["date"].dt.date.isin(fr_holidays)

    # Drop the original 'date' column
    return X


# %%


# %%
date_encoder = FunctionTransformer(_encode_dates, validate=False)
X = date_encoder.fit_transform(X)
X.head(2)

# %%
# Ensure the 'date' column is in datetime format
X["date"] = pd.to_datetime(X["date"])

# Find the earliest and latest dates
earliest_date = X["date"].min()
latest_date = X["date"].max()

print(f"Earliest date: {earliest_date}")
print(f"Latest date: {latest_date}")


# %%
strike_data = {
    "date": [
        datetime(2023, 2, 7),
        datetime(2023, 2, 16),
        datetime(2023, 3, 7),
        datetime(2023, 1, 31),
        datetime(2022, 2, 18),
        datetime(2022, 3, 25),
        datetime(2022, 5, 23),
        datetime(2022, 9, 29),
        datetime(2022, 10, 13),
    ],
    "Strike": [1] * 9,
}

# Create a DataFrame
strike = pd.DataFrame(strike_data)

# Sort the values by ascending date
strike.sort_values(by="date", inplace=True)
strike.reset_index(drop=True, inplace=True)

# Merge the strike DataFrame with df
X = X.merge(strike, on="date", how="left")
X["Strike"] = X["Strike"].fillna(0).astype(int)


# Create get_TimeOfDay_name and get_TimeOfDay functions
def get_TimeOfDay_name(hour):

    if hour > 3 and hour <= 6:
        return "Early morning 4:00AM - 6:00 AM"
    if hour > 6 and hour <= 10:
        return "Morning 7:00AM - 10:00 AM"
    elif hour > 10 and hour <= 13:
        return "Middle of the day 11:00 AM - 1:00 PM"
    elif hour > 13 and hour <= 17:
        return "Afternoon 2:00 PM - 5:00 PM"
    elif hour > 17 and hour <= 22:
        return "Evening 6:00 PM - 10:00 PM"
    else:
        return "Night 11:00 PM - 3:00 AM"


def get_TimeOfDay(hour):
    if hour > 3 and hour <= 6:
        return 1
    if hour > 6 and hour <= 10:
        return 2
    elif hour > 10 and hour <= 13:
        return 3
    elif hour > 13 and hour <= 17:
        return 4
    elif hour > 17 and hour <= 22:
        return 5
    else:
        return 6


# Create columns by applying the functions
X["TimeOfDay"] = X["hour"].apply(get_TimeOfDay)
X["TimeOfDay_name"] = X["hour"].apply(get_TimeOfDay_name)


def get_season_name(date):
    if (date > datetime(2022, 3, 20)) & (date < datetime(2022, 6, 21)):
        return "Spring"
    if (date > datetime(2022, 6, 20)) & (date < datetime(2022, 9, 21)):
        return "Summer"
    if (date > datetime(2022, 9, 20)) & (date < datetime(2022, 12, 21)):
        return "Fall"
    if ((date > datetime(2022, 12, 20)) & (date < datetime(2023, 3, 20))) | (
        (date > datetime(2021, 12, 31)) & (date < datetime(2022, 3, 21))
    ):
        return "Winter"


def get_season(date):
    if (date > datetime(2022, 3, 20)) & (date < datetime(2022, 6, 21)):
        return 1
    if (date > datetime(2022, 6, 20)) & (date < datetime(2022, 9, 21)):
        return 2
    if (date > datetime(2022, 9, 20)) & (date < datetime(2022, 12, 21)):
        return 3
    if ((date > datetime(2022, 12, 20)) & (date < datetime(2023, 3, 20))) | (
        (date > datetime(2021, 12, 31)) & (date < datetime(2022, 3, 21))
    ):
        return 4


# Create columns by applying the functions
X["Season"] = X["date"].apply(get_season)
X["Season_name"] = X["date"].apply(get_season_name)

# %%


# %%
# X = pd.get_dummies(X, columns=["hour"], prefix="hour")
# X.head(2)


# %%
def train_test_split_temporal(X, y, delta_threshold="30 days"):

    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = X["date"] <= cutoff_date
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid


# %%
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# %%
X.dtypes

# %%
datetime_columns = X.select_dtypes(include=["datetime64[ns]"]).columns
print(f"Columns with datetime64[ns] dtype: {datetime_columns.tolist()}")

# %%
# Step 1: Preprocessing
# One-hot encode the categorical variables
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_encoded = onehot_encoder.fit_transform(X[categorical_cols])

# Numerical scaling
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
scaler = MinMaxScaler()
numerical_scaled = scaler.fit_transform(X[numerical_cols])

# Drop processed columns
X.drop(categorical_cols, axis=1, inplace=True)
X.drop(numerical_cols, axis=1, inplace=True)

# Ensure date column is in datetime format
X["date"] = pd.to_datetime(X["date"])

# Combine all features
X_combined = np.hstack([X.values, categorical_encoded, numerical_scaled])

# Step 3: Temporal Train-Test Split
# Convert X_reshaped back into a DataFrame to preserve the date column
X_combined_df = pd.DataFrame(
    X_combined, columns=[f"feature_{i}" for i in range(X_combined.shape[1])]
)
X_combined_df["date"] = X["date"].values  # Restore the date column


# Apply temporal train-test split
X_train_split, y_train_split, X_test_split, y_test_split = train_test_split_temporal(
    X_combined_df, y
)


# Remove the 'date' column after splitting
datetime_columns = X_test_split.select_dtypes(include=["datetime64[ns]"]).columns
print(f"Columns with datetime64[ns] dtype: {datetime_columns.tolist()}")
datetime_columns = X_test_split.select_dtypes(include=["datetime64[ns]"]).columns
print(f"Columns with datetime64[ns] dtype: {datetime_columns.tolist()}")

# Drop these columns from X_train_split
X_train_split = X_train_split.drop(columns=datetime_columns)
# Drop these columns from X_test_split
X_test_split = X_test_split.drop(columns=datetime_columns)

X_train_split = X_train_split.astype(float)
X_test_split = X_test_split.astype(float)


# %%
categorical_cols

# %%
numerical_cols

# %%
datetime_columns

# %%
X_train_split.info()

# %%
final_test = utils.get_test_data()
original_index = final_test.index
date_encoder = FunctionTransformer(_encode_dates, validate=False)
final_test = date_encoder.fit_transform(final_test)
# final_test = pd.get_dummies(final_test, columns=["hour"], prefix="hour")
# final_test.head(2)

# Step 1: Preprocessing
# One-hot encode the categorical variables

final_test = final_test.merge(strike, on="date", how="left")
final_test["Strike"] = final_test["Strike"].fillna(0).astype(int)

final_test["TimeOfDay"] = final_test["hour"].apply(get_TimeOfDay)
final_test["TimeOfDay_name"] = final_test["hour"].apply(get_TimeOfDay_name)

final_test["Season"] = final_test["date"].apply(get_season)
final_test["Season_name"] = final_test["date"].apply(get_season_name)

categorical_cols = final_test.select_dtypes(include=["object", "category"]).columns
onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_encoded = onehot_encoder.fit_transform(final_test[categorical_cols])

# Numerical scaling
numerical_cols = final_test.select_dtypes(include=["int64", "float64"]).columns
scaler = MinMaxScaler()
numerical_scaled = scaler.fit_transform(final_test[numerical_cols])

# Drop processed columns
final_test.drop(categorical_cols, axis=1, inplace=True)
final_test.drop(numerical_cols, axis=1, inplace=True)

# Ensure date column is in datetime format
final_test["date"] = pd.to_datetime(final_test["date"])

# Combine all features
final_test_combined = np.hstack(
    [final_test.values, categorical_encoded, numerical_scaled]
)

# Assuming each sample has a single timestep

# Convert X_reshaped back into a DataFrame to preserve the date column
final_test_combined = pd.DataFrame(
    final_test_combined,
    columns=[f"feature_{i}" for i in range(final_test_combined.shape[1])],
)
final_test_combined["date"] = final_test["date"].values  # Restore the date column


datetime_columns = final_test_combined.select_dtypes(include=["datetime64[ns]"]).columns
print(f"Columns with datetime64[ns] dtype: {datetime_columns.tolist()}")

final_test_combined = final_test_combined.drop(columns=datetime_columns)

final_test_combined = final_test_combined.astype(float)


# %%


# %%
# models = {
#     "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=1,n_estimators= 189, max_depth= 10, learning_rate= 0.09580795720537544, subsample = 0.8622878787892132),
#     "LightGBM": lgb.LGBMRegressor(random_state=42,num_leaves= 100,
#  max_depth= 28,
#  learning_rate=0.12447033793139023,
#  n_estimators=162),
#     # "Random Forest": RandomForestRegressor(
#     #     n_estimators=100,  # Fewer trees
#     #     max_depth=20,     # Limit depth
#     #     min_samples_split=5,
#     #     min_samples_leaf=2,
#     #     random_state=42,
#     #     n_jobs=-1         # Utilize multiple cores
#     # ),
# }

# # Initialize a dictionary to store results
# results = {}

# # Train and evaluate each model
# for name, model in models.items():
#     # Train the model
#     model.fit(X_train_split, y_train_split)

#     # Predict on the test set
#     y_pred = model.predict(X_test_split)

#     # Calculate RMSE
#     rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
#     results[name] = rmse
#     print(f"Model: {name}, RMSE: {rmse}")

# # Convert results to a DataFrame and display
# results_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE']).sort_values(by='RMSE')

# # Display the results
# print(results_df)

# for name, model in models.items():
#     predictions = model.predict(final_test_combined)
#     submission = pd.DataFrame({"id": original_index, "log_bike_count": predictions.flatten()})
#     submission_path = f"submission_{name}.csv"
#     submission.to_csv(submission_path, index=False)

# %%
# Train base models
rf = RandomForestRegressor(
    n_estimators=149,  # Fewer trees
    max_depth=30,  # Limit depth
    min_samples_split=6,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,  # Utilize multiple cores
)
lgb_model = lgb.LGBMRegressor(
    random_state=42,
    num_leaves=183,
    max_depth=10,
    learning_rate=0.10718414123590023,
    n_estimators=179,
    subsample=0.5260664923912034,
    colsample_bytree=0.6560004292557168,
    min_child_samples=74,
    reg_alpha=5.263751804866324,
    reg_lambda=8.783868704821826,
)

rf.fit(X_train_split, y_train_split)
# xgb_model.fit(X_train_split, y_train_split)
lgb_model.fit(X_train_split, y_train_split)

# Generate predictions for stacking
rf_pred = rf.predict(X_test_split)
# xgb_pred = xgb_model.predict(X_test_split)
lgb_pred = lgb_model.predict(X_test_split)

# Combine predictions as input to the meta-model
stacked_features = np.vstack((rf_pred, lgb_pred)).T

# Train meta-model
meta_model = xgb.XGBRegressor(
    random_state=42,
    verbosity=1,
    n_estimators=675,
    max_depth=10,
    learning_rate=0.13518125330035366,
    subsample=0.8205993182340949,
    colsample_bytree=0.5068031953462258,
    colsample_bylevel=0.8644667494430613,
    colsample_bynode=0.7972962473395386,
    gamma=2.545420454736484,
    min_child_weight=6,
    reg_alpha=3.080165279678978,
    reg_lambda=7.357446200895196,
)
meta_model.fit(stacked_features, y_test_split)

# Final predictions
final_pred = meta_model.predict(stacked_features)

# Evaluate the stacked model
rmse = np.sqrt(mean_squared_error(y_test_split, final_pred))
print(f"RMSE of Stacked Model: {rmse}")


# %%
rf_pred = rf.predict(final_test_combined)
lgb_pred = lgb_model.predict(final_test_combined)

# Combine predictions as input to the meta-model
stacked_features = np.vstack((rf_pred, lgb_pred)).T

# Final predictions
predictions = meta_model.predict(stacked_features)

submission = pd.DataFrame(
    {"id": original_index, "log_bike_count": predictions.flatten()}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

# %%
# pip install optuna

# %%
# import optuna

# # Define Optuna objective function for XGBoost
# def xgb_objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
#         "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
#         "random_state": 42,
#         "objective": "reg:squarederror"
#     }
#     model = xgb.XGBRegressor(**params)
#     model.fit(X_train_split, y_train_split)
#     y_pred = model.predict(X_test_split)
#     rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
#     return rmse

# # Define Optuna objective function for LightGBM
# def lgb_objective(trial):
#     params = {
#         "num_leaves": trial.suggest_int("num_leaves", 20, 200),
#         "max_depth": trial.suggest_int("max_depth", -1, 30),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
#         "random_state": 42,
#         "objective": "regression"
#     }
#     model = lgb.LGBMRegressor(**params)
#     model.fit(X_train_split, y_train_split)
#     y_pred = model.predict(X_test_split)
#     rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
#     return rmse

# # Run Optuna for XGBoost
# print("Tuning XGBoost with Optuna...")
# xgb_study = optuna.create_study(direction="minimize")
# xgb_study.optimize(xgb_objective, n_trials=10)
# best_xgb_params = xgb_study.best_params
# print("Best XGBoost Params:", best_xgb_params)

# # Run Optuna for LightGBM
# print("Tuning LightGBM with Optuna...")
# lgb_study = optuna.create_study(direction="minimize")
# lgb_study.optimize(lgb_objective, n_trials=10)
# best_lgb_params = lgb_study.best_params
# print("Best LightGBM Params:", best_lgb_params)

# # Evaluate optimized models
# best_xgb = xgb.XGBRegressor(**best_xgb_params, random_state=42)
# best_xgb.fit(X_train_split, y_train_split)

# best_lgb = lgb.LGBMRegressor(**best_lgb_params, random_state=42)
# best_lgb.fit(X_train_split, y_train_split)

# models = {"XGBoost": best_xgb, "LightGBM": best_lgb}
# results = {}

# for name, model in models.items():
#     y_pred = model.predict(X_test_split)
#     rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
#     results[name] = rmse
#     print(f"Model: {name}, RMSE: {rmse}")

# results_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE']).sort_values(by='RMSE')
# print(results_df)


# %%
# import optuna
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import lightgbm as lgb
# import xgboost as xgb
# import numpy as np

# # Optuna Objective for RandomForest
# def rf_objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 200),
#         "max_depth": trial.suggest_int("max_depth", 5, 30),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
#         "n_jobs": -1,
#         "random_state": 42
#     }
#     rf_model = RandomForestRegressor(**params)
#     rf_model.fit(X_train_split, y_train_split)
#     rf_pred = rf_model.predict(X_test_split)
#     rmse = np.sqrt(mean_squared_error(y_test_split, rf_pred))
#     return rmse

# # Optuna Objective for LightGBM
# def lgb_objective(trial):
#     params = {
#         "num_leaves": trial.suggest_int("num_leaves", 20, 100),
#         "max_depth": trial.suggest_int("max_depth", -1, 30),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "n_estimators": trial.suggest_int("n_estimators", 50, 200),
#         "random_state": 42
#     }
#     lgb_model = lgb.LGBMRegressor(**params)
#     lgb_model.fit(X_train_split, y_train_split)
#     lgb_pred = lgb_model.predict(X_test_split)
#     rmse = np.sqrt(mean_squared_error(y_test_split, lgb_pred))
#     return rmse

# # Optuna Objective for XGBoost (Meta-Model)
# def xgb_objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 200),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "random_state": 42
#     }
#     xgb_model = xgb.XGBRegressor(**params)
#     xgb_model.fit(stacked_features, y_test_split)
#     final_pred = xgb_model.predict(stacked_features)
#     rmse = np.sqrt(mean_squared_error(y_test_split, final_pred))
#     return rmse

# # Run Optuna for RandomForest
# print("Optimizing RandomForest...")
# rf_study = optuna.create_study(direction="minimize")
# rf_study.optimize(rf_objective, n_trials=20)
# best_rf_params = rf_study.best_params
# print("Best RandomForest Params:", best_rf_params)

# # Run Optuna for LightGBM
# print("Optimizing LightGBM...")
# lgb_study = optuna.create_study(direction="minimize")
# lgb_study.optimize(lgb_objective, n_trials=50)
# best_lgb_params = lgb_study.best_params
# print("Best LightGBM Params:", best_lgb_params)

# # Train base models with optimized hyperparameters
# best_rf = RandomForestRegressor(**best_rf_params, random_state=42)
# best_rf.fit(X_train_split, y_train_split)
# rf_pred = best_rf.predict(X_test_split)

# best_lgb = lgb.LGBMRegressor(**best_lgb_params, random_state=42)
# best_lgb.fit(X_train_split, y_train_split)
# lgb_pred = best_lgb.predict(X_test_split)

# # Combine predictions as input to meta-model
# stacked_features = np.vstack((rf_pred, lgb_pred)).T

# # Run Optuna for XGBoost
# print("Optimizing Meta-Model (XGBoost)...")
# xgb_study = optuna.create_study(direction="minimize")
# xgb_study.optimize(xgb_objective, n_trials=50)
# best_xgb_params = xgb_study.best_params
# print("Best XGBoost Params:", best_xgb_params)

# # Train meta-model with optimized hyperparameters
# best_meta_model = xgb.XGBRegressor(**best_xgb_params, random_state=42)
# best_meta_model.fit(stacked_features, y_test_split)

# # Final predictions with the stacked model
# final_pred = best_meta_model.predict(stacked_features)

# # Evaluate the stacked model
# rmse = np.sqrt(mean_squared_error(y_test_split, final_pred))
# print(f"Optimized RMSE of Stacked Model: {rmse}")


# %%
# best_lgb_params

# %%
# best_rf_params

# %%
