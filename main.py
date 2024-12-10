# %%
import subprocess

# subprocess.run([
#     "latitude", "longitude", "year", "month", "day", "weekday", "hour",
#     "is_weekend", "is_holiday", "strike", "lockdown", "TimeOfDay", "Season"
# ])
import data_cleaning

# %%
import optuna
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# %%
problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"

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
erase_date = FunctionTransformer(data_cleaning.erase_date)

ordinal_cols = ["counter_installation_date", "counter_id"]
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
]

scaler = StandardScaler()
ordinal = OrdinalEncoder()

preprocessor = ColumnTransformer(
    [
        ("num", scaler, scale_cols),
        ("ordinal", ordinal, ordinal_cols),
    ]
)

regressor = XGBRegressor(random_state=42)
# Create the pipeline
pipe = make_pipeline(
    columns_encoder,
    date_encoder,
    strike_encoder,
    lockdown_encoder,
    time_of_day_encoder,
    season_encoder,
    erase_date,
    preprocessor,
    regressor,
)

# Fit the pipeline
pipe.fit(X_train_split, y_train_split)

# Evaluate the model
y_pred = pipe.predict(X_test_split)
rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
print(rmse)

# %%
final_test = get_test_data()
original_index = final_test.index

# %%
y_pred = pipe.predict(final_test)
results = pd.DataFrame(
    dict(
        Id=original_index,
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)


# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


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
    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
)
lgb_model = LGBMRegressor(random_state=42)

# Meta-model
xgb_meta_model = XGBRegressor(random_state=42)

# Define encoders and preprocessors
columns_encoder = FunctionTransformer(data_cleaning._select_columns)
date_encoder = FunctionTransformer(data_cleaning._encode_dates)
strike_encoder = FunctionTransformer(data_cleaning._add_strike)
lockdown_encoder = FunctionTransformer(data_cleaning._add_lockdown)
time_of_day_encoder = FunctionTransformer(data_cleaning._add_time_of_day)
season_encoder = FunctionTransformer(data_cleaning._add_season)
erase_date = FunctionTransformer(data_cleaning.erase_date)

ordinal_cols = ["counter_installation_date", "counter_id"]
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
]

scaler = StandardScaler()
ordinal = OrdinalEncoder()

preprocessor = ColumnTransformer(
    [
        ("num", scaler, scale_cols),
        ("ordinal", ordinal, ordinal_cols),
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
final_test = data_cleaning.get_test_data(path="data/final_test.parquet")
original_index = final_test.index
final_test_predictions = stacking_pipeline.predict(final_test)

# Create a submission file
submission = pd.DataFrame(
    {"id": original_index, "log_bike_count": final_test_predictions.flatten()}
)
submission_path = "submission_stacked_pipeline.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved at: {submission_path}")


# %%
