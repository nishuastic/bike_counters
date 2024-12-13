import subprocess
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna
import data_cleaning  # Ensure this module contains all preprocessing functions

# Load data
X, y = data_cleaning.get_train_data(path="data/train.parquet")
X_train_split, y_train_split, X_test_split, y_test_split = (
    data_cleaning.train_test_split_temporal(X, y)
)

# Define encoders and preprocessors
columns_encoder = FunctionTransformer(data_cleaning._select_columns)
date_encoder = FunctionTransformer(data_cleaning._encode_dates)
strike_encoder = FunctionTransformer(data_cleaning._add_strike)
lockdown_encoder = FunctionTransformer(data_cleaning._add_lockdown)
time_of_day_encoder = FunctionTransformer(data_cleaning._add_time_of_day)
season_encoder = FunctionTransformer(data_cleaning._add_season)
district_encoder = FunctionTransformer(data_cleaning._add_district_name)
weather_data_encoder = FunctionTransformer(data_cleaning._merge_weather_data)
erase_date = FunctionTransformer(data_cleaning._erase_date)

onehot_cols = [
    "TimeOfDay",
    "Season",
    "is_weekend",
    "is_holiday",
    "strike",
    "lockdown",
]
scale_cols = [
    "latitude",
    "longitude",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "t",
    "ff",
    "pres",
    # "ssfrai",
    # "ht_neige",
    # "rr1",
    # "rr3",
    # "rr6",
    # "rr12",
    # "rr24",
    "vv",
    # "n",
    # "u"
]
ordinal_cols = [
    "year",
    # "month",
    "day",
    # "weekday",
    # "hour",
    "counter_installation_date",
]

scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ordinal = OrdinalEncoder()
preprocessor = ColumnTransformer(
    [
        ("num", scaler, scale_cols),
        ("onehot", onehot, onehot_cols),
        ("ordinal", ordinal, ordinal_cols),
    ]
)

preprocessing_pipeline = Pipeline(
    steps=[
        ("columns_encoder", columns_encoder),
        ("date_encoder", date_encoder),
        ("strike_encoder", strike_encoder),
        ("lockdown_encoder", lockdown_encoder),
        ("time_of_day_encoder", time_of_day_encoder),
        ("season_encoder", season_encoder),
        # ("district_encoder", district_encoder),
        ("weather_data_encoder", weather_data_encoder),
        ("erase_date", erase_date),
        ("preprocessor", preprocessor),
    ]
)

# Preprocess the data
X_train_preprocessed = preprocessing_pipeline.fit_transform(
    X_train_split, y_train_split
)
X_test_preprocessed = preprocessing_pipeline.transform(X_test_split)


# Step 1: Optimize Random Forest
def optimize_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train_preprocessed, y_train_split)
    y_pred = model.predict(X_test_preprocessed)
    rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
    return rmse


print("Optimizing Random Forest...")
rf_study = optuna.create_study(direction="minimize")
rf_study.optimize(optimize_rf, n_trials=15)
best_rf_params = rf_study.best_params
print(f"Best Random Forest Params: {best_rf_params}")

# Train optimized Random Forest
rf_model = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train_preprocessed, y_train_split)
rf_train_pred = rf_model.predict(X_train_preprocessed)
rf_test_pred = rf_model.predict(X_test_preprocessed)


# Step 2: Optimize LightGBM
def optimize_lgb(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 10.0),
    }
    model = LGBMRegressor(**params, random_state=42)
    model.fit(X_train_preprocessed, y_train_split)
    y_pred = model.predict(X_test_preprocessed)
    rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
    return rmse


print("Optimizing LightGBM...")
lgb_study = optuna.create_study(direction="minimize")
lgb_study.optimize(optimize_lgb, n_trials=25)
best_lgb_params = lgb_study.best_params
print(f"Best LightGBM Params: {best_lgb_params}")

# Train optimized LightGBM
lgb_model = LGBMRegressor(**best_lgb_params, random_state=42)
lgb_model.fit(X_train_preprocessed, y_train_split)
lgb_train_pred = lgb_model.predict(X_train_preprocessed)
lgb_test_pred = lgb_model.predict(X_test_preprocessed)


# Step 3: Optimize XGBoost (Meta-Model)
stacked_train_features = np.vstack((rf_train_pred, lgb_train_pred)).T
stacked_test_features = np.vstack((rf_test_pred, lgb_test_pred)).T


def optimize_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_uniform("colsample_bynode", 0.5, 1.0),
        "gamma": trial.suggest_loguniform("gamma", 1e-5, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 10.0),
    }
    model = XGBRegressor(**params, random_state=42)
    model.fit(stacked_train_features, y_train_split)
    y_pred = model.predict(stacked_test_features)
    rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
    return rmse


print("Optimizing XGBoost...")
xgb_study = optuna.create_study(direction="minimize")
xgb_study.optimize(optimize_xgb, n_trials=25)
best_xgb_params = xgb_study.best_params
print(f"Best XGBoost Params: {best_xgb_params}")

# Train optimized XGBoost
xgb_meta_model = XGBRegressor(**best_xgb_params, random_state=42)
xgb_meta_model.fit(stacked_train_features, y_train_split)

# Final Evaluation
final_pred = xgb_meta_model.predict(stacked_test_features)
final_rmse = np.sqrt(mean_squared_error(y_test_split, final_pred))
print(f"Final RMSE of Stacked Model: {final_rmse:.5f}")

# Final Test Prediction
final_test = data_cleaning.get_test_data(path="data/final_test.parquet")
final_test_preprocessed = preprocessing_pipeline.transform(final_test)
rf_test_final = rf_model.predict(final_test_preprocessed)
lgb_test_final = lgb_model.predict(final_test_preprocessed)
stacked_final_features = np.vstack((rf_test_final, lgb_test_final)).T
final_test_predictions = xgb_meta_model.predict(stacked_final_features)

# Submission File
submission = pd.DataFrame(
    {"id": final_test.index, "log_bike_count": final_test_predictions.flatten()}
)
submission.to_csv("submission.csv", index=False)
print("Submission file saved: submission.csv")
print(f"Best Random Forest Params: {best_rf_params}")
print(f"Best LightGBM Params: {best_lgb_params}")
print(f"Best XGBoost Params: {best_xgb_params}")
