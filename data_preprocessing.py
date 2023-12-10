import pandas as pd
import numpy as np
from constants import electricity_cutoff_threshold, temperature_cutoff_threshold
import logging


def load_data(path_flow: str, path_temperature: str) -> (pd.DataFrame, pd.DataFrame):
    """Function that does the data loading.
    Params: path_flow - String to path of electricity_flow.csv
            path_temperature - String to path of inside_temperature.csv
    Returns: (electricity_flow_df, inside_temperature_df)
    """
    electricity_flow_df = pd.read_csv(path_flow)
    inside_temperature_df = pd.read_csv(path_temperature)
    return electricity_flow_df, inside_temperature_df


def remove_outliers(df: pd.DataFrame, col: str, threshold: int) -> pd.DataFrame:
    """Function that removes outliers based on threshold and fills those values with interpolation.
    Params: df - pd.DataFrame input dataframe
            col - String with column to be cleaned from outliers
            threshold - value with cutoff threshold for outliers
    Returns: pd.DataFrame with outliers removed
    """
    outlier_idx = df.index[df[col] >= threshold]
    df.loc[outlier_idx, col] = np.nan
    df[col] = df[col].interpolate()
    return df


def add_year_month_day(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df["year"] = df[time_col].dt.year
    df["month"] = df[time_col].dt.month
    df["day"] = df[time_col].dt.day
    df["hour"] = df[time_col].dt.hour
    return df


def create_customer_df(path_flow: str, path_temperature: str) -> pd.DataFrame:
    """Function that creates the customer_df. Does loading and preprocessing of the data
    before merging them into one dataframe.
    Params: path_flow - str
            path_temperature: str
    Returns: customer_df - pd.DataFrame sorted, without outliers for temperature and electricity in minute granularity.
    """
    # Load data and make sure values are sorted per user and time
    logging.info("Loading data...")
    electricity_flow_df, inside_temperature_df = load_data(path_flow, path_temperature)
    electricity_flow_df["utc_datetime"] = pd.to_datetime(
        electricity_flow_df["utc_datetime"]
    )
    inside_temperature_df["utc_datetime"] = pd.to_datetime(
        inside_temperature_df["utc_datetime"]
    )
    electricity_flow_df = electricity_flow_df.sort_values(by=["user", "utc_datetime"])
    inside_temperature_df = inside_temperature_df.sort_values(
        by=["user", "utc_datetime"]
    )

    # Remove outliers from dataframes
    logging.info("Removing outliers...")
    electricity_flow_df = remove_outliers(
        electricity_flow_df, "electricity", electricity_cutoff_threshold
    )
    inside_temperature_df = remove_outliers(
        inside_temperature_df, "inside", temperature_cutoff_threshold
    )

    # Join dataframes on user and time, left join since electricity_flow_df has more time gaps than inside_temperature_df
    customer_df = inside_temperature_df.merge(
        electricity_flow_df, how="left", on=["user", "utc_datetime"]
    )
    customer_df = add_year_month_day(customer_df, "utc_datetime")
    return customer_df
