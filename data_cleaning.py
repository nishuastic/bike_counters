# %%
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import holidays
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

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


def get_train_data(path="/kaggle/input/msdb-2024/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def get_test_data(path="/kaggle/input/msdb-2024/final_test.parquet"):
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
    X["is_holiday"] = X["date"].dt.date.isin(fr_holidays)

    X['hour_sin'] = np.sin(2 * np.pi * X['hour']/24)
    X['hour_cos'] = np.cos(2 * np.pi * X['hour']/24)
    X['day_of_week_sin'] = np.sin(2 * np.pi * X['weekday']/7)
    X['day_of_week_cos'] = np.cos(2 * np.pi * X['weekday']/7)
    X['month_sin'] = np.sin(2 * np.pi * X['month']/12)
    X['month_cos'] = np.cos(2 * np.pi * X['month']/12)

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

    def get_season(date):
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

    X["Season"] = X["date"].apply(get_season)
    return X


def _add_district_name(X, geojson_path="/kaggle/input/msdb-2024/arrondissements.geojson"):

    arrondissements = gpd.read_file(geojson_path)

    X = X.copy()
    X["geometry"] = X.apply(
        lambda row: Point(row["longitude"], row["latitude"]), axis=1
    )
    X_geo = gpd.GeoDataFrame(X, geometry="geometry", crs=arrondissements.crs)

    # Perform a spatial join to map latitude/longitude to arrondissement
    X_geo = gpd.sjoin(X_geo, arrondissements, how="left", predicate="within")

    # Add arrondissement name column
    X["district"] = X_geo["l_aroff"].fillna("Unknown")  # Replace NaN with "Unknown"

    # Drop the geometry column if not needed
    X.drop(columns=["geometry"], inplace=True)

    return X


def _merge_weather_data(X, weather_df_path="data/external_data.csv"):

    X = X.copy()

    # Load and preprocess weather data
    weather_df = pd.read_csv(weather_df_path)
    weather_df["date"] = pd.to_datetime(weather_df["date"]).astype("datetime64[ns]")
    X["date"] = pd.to_datetime(X["date"]).astype("datetime64[ns]")

    # Preserve original index
    X["orig_index"] = np.arange(X.shape[0])

    # Perform merge_asof
    X = pd.merge_asof(
        X.sort_values("date"),
        weather_df[
            [
                "date",
                "ff",
                "pres",
                "ssfrai",
                "ht_neige",
                "rr1",
                "rr3",
                "rr6",
                "rr12",
                "rr24",
                "vv",
                "ww",
                "n",
                "t",
            ]
        ].sort_values("date"),
        on="date",
    )

    # Restore the original order and clean up
    X = X.sort_values("orig_index")
    del X["orig_index"]

    # Fill missing values with specific logic
    X[["ssfrai", "ht_neige"]] = X[["ssfrai", "ht_neige"]].fillna(0)
    X[["rr1", "rr3", "rr6", "rr12", "rr24", "vv"]] = X[
        ["rr1", "rr3", "rr6", "rr12", "rr24", "vv"]
    ].fillna(X[["rr1", "rr3", "rr6", "rr12", "rr24", "vv"]].mean())

    if X["n"].isna().any():
        mean_value = X["n"].mean(skipna=True)
        rounded_mean = round(mean_value)
        X["n"].fillna(rounded_mean, inplace=True)

    return X

def _erase_date(X):
    X = X.copy()
    datetime_columns = X.select_dtypes(include=["datetime64"]).columns
    X = X.drop(columns=datetime_columns)
    return X

# %%
