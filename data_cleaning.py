# %%
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import holidays
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# %%
# Functions from utils.py
problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def train_test_split_temporal(X, y, delta_threshold="30 days"):

    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = X["date"] <= cutoff_date
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def get_test_data(path="data/final_test.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    return data


def _select_columns(X):
    X = X.copy()
    columns_to_drop = [
        "counter_name",
        "site_id",
        "site_name",
        "coordinates",
        "counter_technical_id",
    ]
    X = X.drop(columns=columns_to_drop, axis=1)
    return X


def _encode_dates(X):
    X = X.copy()

    X["date"] = pd.to_datetime(X["date"])

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

    return X


def _add_strike(X):
    X = X.copy()

    strike_data = {
        "date": [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 7),
            datetime(2020, 1, 8),
            datetime(2020, 1, 9),
            datetime(2020, 1, 10),
            datetime(2020, 10, 17),
            datetime(2020, 10, 18),
            datetime(2021, 2, 4),
            datetime(2021, 2, 5),
            datetime(2021, 4, 6),
            datetime(2021, 4, 7),
            datetime(2021, 11, 16),
            datetime(2021, 11, 17),
        ],
        "strike": [1] * 16,
    }

    # Create DataFrame for strikes
    strike = pd.DataFrame(strike_data)

    # Merge with the input DataFrame
    X = X.merge(strike, on="date", how="left")
    X["strike"] = X["strike"].fillna(0).astype(int)

    return X


def _add_lockdown(X):
    X = X.copy()

    lockdown_periods = [
        (pd.Timestamp("2020-03-17"), pd.Timestamp("2020-05-11")),
        (pd.Timestamp("2020-10-30"), pd.Timestamp("2020-12-15")),
        (pd.Timestamp("2021-04-03"), pd.Timestamp("2021-05-03")),
    ]

    # Function to check if a date is in a lockdown period
    def is_lockdown(date):
        for start, end in lockdown_periods:
            if start <= date <= end:
                return 1
        return 0

    # Apply lockdown logic
    X["lockdown"] = X["date"].apply(is_lockdown)

    return X


def _add_time_of_day(X):
    X = X.copy()

    # Function to categorize hours into time of day
    def get_time_of_day(hour):
        if hour > 3 and hour <= 6:
            return 1  # Early Morning
        if hour > 6 and hour <= 10:
            return 2  # Morning
        elif hour > 10 and hour <= 13:
            return 3  # Midday
        elif hour > 13 and hour <= 17:
            return 4  # Afternoon
        elif hour > 17 and hour <= 22:
            return 5  # Evening
        else:
            return 6  # Night

    # Apply the function to the 'hour' column
    X["TimeOfDay"] = X["hour"].apply(get_time_of_day)

    return X


def _add_season(X):
    X = X.copy()

    # Function to assign seasons for 2020 and 2021
    def get_season_label(date):
        if ((date > datetime(2020, 3, 20)) & (date < datetime(2020, 6, 21))) | (
            (date > datetime(2021, 3, 20)) & (date < datetime(2021, 6, 21))
        ):
            return 1  # Spring
        if ((date > datetime(2020, 6, 20)) & (date < datetime(2020, 9, 21))) | (
            (date > datetime(2021, 6, 20)) & (date < datetime(2021, 9, 21))
        ):
            return 2  # Summer
        if ((date > datetime(2020, 9, 20)) & (date < datetime(2020, 12, 21))) | (
            (date > datetime(2021, 9, 20)) & (date < datetime(2021, 12, 21))
        ):
            return 3  # Fall
        if (
            ((date > datetime(2020, 12, 20)) & (date < datetime(2021, 3, 21)))
            | ((date > datetime(2019, 12, 31)) & (date < datetime(2020, 3, 21)))
            | ((date > datetime(2021, 12, 20)) & (date < datetime(2022, 3, 21)))
        ):
            return 4  # Winter

    X["Season"] = X["date"].apply(get_season_label)

    return X


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t"]].sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def erase_date(X):
    X = X.copy()
    return X.drop("date", axis=1)


# %%
