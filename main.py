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
        "geopandas",
        "shapely",
        "holidays",
    ]
)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

import sys

sys.path.append("/kaggle/usr/lib/data_cleaning/")
import data_cleaning

X, y = data_cleaning.get_train_data(path="/kaggle/input/msdb-2024/train.parquet")

X_train_split, y_train_split, X_test_split, y_test_split = (
    data_cleaning.train_test_split_temporal(X, y)
)


# Custom transformer for stacking
class StackingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rf_model, lgb_model):
        self.rf_model = rf_model
        self.lgb_model = lgb_model

    def fit(self, X, y):
        # Fit base models on training data
        self.rf_model.fit(X, y)
        self.lgb_model.fit(X, y)
        return self

    def transform(self, X):
        # Generate predictions from base models
        rf_pred = self.rf_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        # Combine predictions into stacked features
        return np.vstack((rf_pred, lgb_pred)).T


# Base models
rf_model = RandomForestRegressor(
    n_estimators=90,
    max_depth=27,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
lgb_model = LGBMRegressor(
    random_state=1,
    num_leaves=192,
    max_depth=13,
    learning_rate=0.07436527963995643,
    n_estimators=475,
    subsample=0.5080506284836919,
    colsample_bytree=0.5728392728761649,
    min_child_samples=53,
    reg_alpha=0.5940178261627549,
    reg_lambda=3.4178326150733502,
)

# Meta-model
xgb_meta_model = XGBRegressor(
    random_state=42,
    verbosity=1,
    n_estimators=425,
    max_depth=15,
    learning_rate=0.010243232775566815,
    subsample=0.8611241400339762,
    colsample_bytree=0.6339637458866162,
    colsample_bylevel=0.6517707978638164,
    colsample_bynode=0.6457651375679931,
    gamma=0.0003063698929298127,
    min_child_weight=5,
    reg_alpha=0.0029023896794441806,
    reg_lambda=3.955647179874124e-05,
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

onehot_cols = ["counter_id", "ww", "district"]
scale_cols = [
    "latitude",
    "longitude",
    "year",
    "month",
    "day",
    "weekday",
    "hour",
    "is_weekend",
    "is_holiday",
    "strike",
    "lockdown",
    "TimeOfDay",
    "Season",
    "ff",
    "pres",
    "ssfrai",
    "ht_neige",
    "rr1",
    "rr3",
    "rr6",
    "rr12",
    "vv",
    "n",
    "t",
]
ordinal_cols = ["counter_installation_date"]

scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


preprocessor = ColumnTransformer(
    [
        ("num", scaler, scale_cols),
        ("onehot", onehot, onehot_cols),
    ]
)

# Pipeline for stacking
stacking_pipeline = Pipeline(
    steps=[
        ("columns_encoder", columns_encoder),
        ("date_encoder", date_encoder),
        ("strike_encoder", strike_encoder),
        ("lockdown_encoder", lockdown_encoder),
        ("time_of_day_encoder", time_of_day_encoder),
        ("season_encoder", season_encoder),
        ("district_encoder", district_encoder),
        ("weather_data_encoder", weather_data_encoder),
        ("erase_date", erase_date),
        ("preprocessor", preprocessor),
        ("stacking", StackingTransformer(rf_model=rf_model, lgb_model=lgb_model)),
        ("meta_model", xgb_meta_model),
    ]
)

# Train the pipeline
stacking_pipeline.fit(X_train_split, y_train_split)

# Evaluate the pipeline
y_pred = stacking_pipeline.predict(X_test_split)
rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
print(f"RMSE of Stacking Pipeline: {rmse:.5f}")

# Predict on final test set
final_test = data_cleaning.get_test_data(
    path="/kaggle/input/msdb-2024/final_test.parquet"
)
original_index = final_test.index
final_test_predictions = stacking_pipeline.predict(final_test)

# Create a submission file
submission = pd.DataFrame(
    {"id": original_index, "log_bike_count": final_test_predictions.flatten()}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
