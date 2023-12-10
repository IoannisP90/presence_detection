import pandas as pd
import numpy as np
from bisect import bisect
from data_preprocessing import create_customer_df
from typing import List
import logging


def create_quantiles_for_time_horizon(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Create quantile values for the user for inside temp and electricity.
    Those will be used to determine the probability of house presence based on his hourly metrics.
    Params: df - pd.DataFrame with user and hourly inside temperature and electricity
            time_col: String with the datetime column
    Returns: pd.DataFrame including the quantile values in a list for every hour for each user
    """
    horizon_user_quantiles = (
        df[["user", time_col, "inside", "electricity"]]
        .groupby(by=["user", time_col])
        .quantile(np.linspace(0, 1, 21))
        .reset_index()
    )
    horizon_user_quantiles = (
        horizon_user_quantiles.groupby(by=["user", time_col])
        .agg(lambda x: x.to_list())
        .reset_index()
    )
    return horizon_user_quantiles


def find_le_idx(num: float, lst: List[float], quantile: List[float]) -> pd.Series:
    return quantile[bisect(lst, num) - 1]


def calculate_partial_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Function that calculates partial probabilities based on the quantile value for temperature and electricity.
    Params: df - (pd.DataFrame) with inside temperature and electricity per hour for every user,
                as well as the quantiles of hourly and part_of_day values for that user.
    Returns: (pd.DataFrame) including new partial probability columns for home presence.
    """
    df["prob_inside_hour"] = df.apply(
        lambda x: find_le_idx(x["inside"], x["inside_hourly_mean"], x["quantiles"]),
        axis=1,
    )
    df["prob_inside_day_part"] = df.apply(
        lambda x: find_le_idx(x["inside"], x["inside_day_part_mean"], x["quantiles"]),
        axis=1,
    )
    df["prob_electricity_hour"] = df.apply(
        lambda x: find_le_idx(
            x["electricity"], x["electricity_hourly_mean"], x["quantiles"]
        ),
        axis=1,
    )
    df["prob_electricity_day_part"] = df.apply(
        lambda x: find_le_idx(
            x["electricity"], x["electricity_day_part_mean"], x["quantiles"]
        ),
        axis=1,
    )
    return df


def create_user_probabilities(path_flow: str, path_temperature: str) -> pd.DataFrame:
    """Function that creates DataFrame with presence probabilities per hour for every user
    "Params": path_flow - string with location of the electricity_flow.csv
              path_temperature - String with path to the inside_temperature.csv
    "Returns": pd.DataFrame with resence probabilities per hour (24 values) for every user
    """
    customer_df = create_customer_df(path_flow, path_temperature)
    # Aggregate hourly mean values for inside temperature and electricity
    resampled_df = (
        customer_df.groupby(by=["user", "month", "day", "hour"])
        .agg({"inside": "mean", "electricity": "mean"})
        .reset_index()
    )
    resampled_df["part_of_day"] = resampled_df["hour"] // 6

    # Create quantile values in steps of 0.05 for temperature and electricity values per user
    logging.info("Creating quantile values...")
    hourly_user_quantiles = create_quantiles_for_time_horizon(resampled_df, "hour")
    day_part_user_quantiles = create_quantiles_for_time_horizon(
        resampled_df, "part_of_day"
    )
    day_part_user_quantiles["part_of_day"] = day_part_user_quantiles[
        "part_of_day"
    ].astype("Int64")

    # Join quantile values per user, per hour
    resampled_df = resampled_df.merge(
        hourly_user_quantiles,
        how="left",
        on=["user", "hour"],
        suffixes=("", "_hourly_mean"),
    )
    resampled_df = resampled_df.merge(
        day_part_user_quantiles,
        how="left",
        on=["user", "part_of_day"],
        suffixes=("", "_day_part_mean"),
    )
    resampled_df = resampled_df.rename(columns={"level_2": "quantiles"}).drop(
        "level_2_day_part_mean", axis=1
    )

    # Calculate the partial presence probabilities for temperature, electricity for hour and part_of_day
    # depending on which hourly value corrseponds to the respective quantile
    logging.info("Calculating user probabilites per hour...")
    resampled_df = calculate_partial_probabilities(resampled_df)
    # Probability of presence is the average of all probabilities
    # ToDo: max of the 4 might be a better predictor
    resampled_df["prob_presence"] = (
        resampled_df["prob_inside_hour"]
        + resampled_df["prob_inside_day_part"]
        + resampled_df["prob_electricity_hour"]
        + resampled_df["prob_electricity_day_part"]
    ) / 4

    probability_df = (
        resampled_df.groupby(by=["user", "hour"])
        .agg({"prob_presence": "mean"})
        .reset_index()
    )
    return probability_df
