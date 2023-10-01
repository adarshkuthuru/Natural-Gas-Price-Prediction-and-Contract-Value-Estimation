#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import Dependencies
import pandas as pd
import numpy as np
from datetime import date, timedelta


def load_data(file_path):
    """
    Load the data from a CSV file, convert 'Dates' to datetime format,
    and make it timezone-naive if required.

    Parameters:
    :param file_path: str - Full path to the CSV file

    Returns:
    :return: data: pd.DataFrame - Loaded and cleaned data
    """
    # Load data from CSV file
    data = pd.read_csv(file_path)

    # Convert 'Dates' column to datetime format
    data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')
    
    # Remove timezone info if it exists
    if data['Dates'].dt.tz is not None:
        data['Dates'] = data['Dates'].dt.tz_localize(None)

    # Convert date column type from timestamp to date
    data['Dates'] = data['Dates'].dt.date

    prices = data['Prices'].values
    dates = data['Dates'].values
    
    return prices, dates


def get_regression_params(prices, dates, start_date):
    """
    Calculate the regression parameters for the provided data.

    Parameters:
    :param prices: np.array - Array of prices
    :param dates: np.array - Array of dates
    :param start_date: date - Start date in date format

    Returns:
    :return: Tuple containing amplitude, shift, slope, days_from_start and intercept values.
    """
    days_from_start = [(day - start_date).days for day in dates]
    time = np.array(days_from_start)
    xbar = np.mean(time)
    ybar = np.mean(prices)
    slope = np.sum((time - xbar) * (prices - ybar)) / np.sum((time - xbar) ** 2)
    intercept = ybar - slope * xbar

    sin_prices = prices - (time * slope + intercept)
    sin_time = np.sin(time * 2 * np.pi / 365)
    cos_time = np.cos(time * 2 * np.pi / 365)
    slope1 = np.sum(sin_prices * sin_time) / np.sum(sin_time ** 2)
    slope2 = np.sum(sin_prices * cos_time) / np.sum(cos_time ** 2)

    amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
    shift = np.arctan2(slope2, slope1)

    return amplitude, shift, slope, intercept, days_from_start



def interpolate_price(input_date, amplitude, shift, slope, intercept, start_date, prices, days_from_start):
    """
    Interpolate or extrapolate prices based on the sine/cos model.

    Parameters:
    :param input_date: str - Date for which to predict the price

    Returns:
    :return: float - Predicted price for the given date
    """
    days = (pd.Timestamp(input_date) - pd.Timestamp(start_date)).days
    if days in days_from_start:
        return prices[days_from_start.index(days)]
    else:
        return amplitude * np.sin(days * 2 * np.pi / 365 + shift) + days * slope + intercept


def contract_value(injection_dates, withdrawal_dates, inject_rate, max_volume,
                   storage_cost_per_day, injection_cost, withdrawal_cost, interpolation_func, *interp_args):
    """
    Calculate the value of a gas contract based on various parameters and the sine/cos model.

    Parameters:
    :param injection_dates: List of str - Dates when gas is injected
    :param withdrawal_dates: List of str - Dates when gas is withdrawn
    :param inject_rate: float - Rate of injection in MMBtu per day
    :param max_volume: float - Maximum storage capacity in MMBtu
    :param storage_cost_per_day: float - Cost of storing gas per day for 1 MMBtu
    :param injection_cost: float - Cost of injecting 1 million MMBtu
    :param withdrawal_cost: float - Cost of withdrawing 1 million MMBtu
    :param interpolation_func: function - Function to interpolate prices
    :param *interp_args: unpacked tuple - Arguments for the interpolation function

    Returns:
    :return: float - Value of the contract in dollars
    """
    assert len(injection_dates) == len(withdrawal_dates), "Mismatched injection and withdrawal dates"
    total_value = 0

    for i in range(len(injection_dates)):
        purchase_price = interpolation_func(injection_dates[i], *interp_args)
        selling_price = interpolation_func(withdrawal_dates[i], *interp_args)
        days_stored = (pd.Timestamp(withdrawal_dates[i]) - pd.Timestamp(injection_dates[i])).days
        volume = min(inject_rate * days_stored, max_volume)
        total_storage_cost = storage_cost_per_day * days_stored * volume
        value = ((selling_price - purchase_price - total_storage_cost - injection_cost - withdrawal_cost) * volume)
        total_value += value

    return total_value

if __name__ == '__main__':

    # Data path
    data_path = r"C:\Users\XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\Nat_Gas.csv"
    
    # Load data and get regression parameters
    prices, dates = load_data(data_path)
    start_date = date(2020, 10, 31)
    amplitude, shift, slope, intercept, days_from_start = get_regression_params(prices, dates, start_date)

    # Define contract parameters and compute its value
    injection_dates = ['2023-06-01', '2023-06-15']
    withdrawal_dates = ['2023-12-01', '2023-12-15']
    inject_rate = 100000  # MMBtu per day
    max_volume = 1e6  # Maximum storage capacity in MMBtu
    storage_cost_per_day = 1000 / 1e6  # $1000 per day for 1 million MMBtu
    injection_cost = 10e3  # $10K per 1 million MMBtu
    withdrawal_cost = 10e3  # $10K per 1 million MMBtu

    value = contract_value(injection_dates, withdrawal_dates, inject_rate, max_volume, 
                           storage_cost_per_day, injection_cost, withdrawal_cost, 
                           interpolate_price, amplitude, shift, slope, intercept, start_date, dates, prices)
    print(f"Value of the contract: ${value:.2f}")