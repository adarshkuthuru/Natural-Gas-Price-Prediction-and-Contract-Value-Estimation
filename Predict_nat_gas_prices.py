#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import Dependencies
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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
    
    return data


def exponential_smoothing_forecast(data):
    """
    Forecast natural gas prices using Exponential Smoothing.

    Parameters:
    :param data: pd.DataFrame - Data containing dates and prices

    Returns:
    :return: forecast: np.array - Predicted values
    :return: forecast_dates: pd.DatetimeIndex - Forecasted dates
    """
    # Apply Exponential Smoothing model
    model = ExponentialSmoothing(data['Prices'], trend='add',
                                 seasonal='add', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(steps=12)
    forecast_dates = pd.date_range(start="2024-10-31", periods=12, freq='M', tz=None)

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data['Dates'], data['Prices'], marker='o', label='Actual Prices')
    plt.plot(forecast_dates, forecast, color='red', marker='o', label='Forecasted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Natural Gas Prices and Forecast Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    return forecast, forecast_dates


def arima_forecast(data):
    """
    Forecast natural gas prices using ARIMA.

    Parameters:
    :param data: pd.DataFrame - Data containing dates and prices

    Returns:
    :return: forecast_arima: np.array - Predicted values
    """
    print("\nForecasting with ARIMA...")
    
    # Apply Auto ARIMA to find best parameters and forecast
    model_arima = auto_arima(data['Prices'], seasonal=True, m=12, trace=True,
                             error_action='ignore', suppress_warnings=True)
    model_arima.fit(data['Prices'])
    forecast_arima = model_arima.predict(n_periods=12)

    # Plot the ARIMA forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data['Dates'], data['Prices'], label='Actual Prices')
    plt.plot(forecast_dates, forecast_arima, color='pink', label='ARIMA Forecast')
    plt.legend()
    plt.show()

    return forecast_arima


def random_forest_forecast(data):
    """Forecast using Random Forest Model.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing 'Dates' and 'Prices' columns
    
    Returns:
    - forecast_rf (list): List of forecasted values
    """

    print("\nForecasting with Random Forest...")

    # Create lag features
    data_rf = data.copy()
    for i in range(1, 13):
        data_rf[f'lag_{i}'] = data_rf['Prices'].shift(i)
    data_rf.dropna(inplace=True)

    X = data_rf.drop(['Prices', 'Dates'], axis=1)
    y = data_rf['Prices']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model_rf = RandomForestRegressor(n_estimators=100)
    model_rf.fit(X_train, y_train)

    # Evaluate model
    y_pred = model_rf.predict(X_test)
    print(f"Random Forest MSE: {mean_squared_error(y_test, y_pred)}")

    # Predicting next 12 months using a recursive strategy
    forecast_rf = []
    last_values_list = list(data['Prices'].tail(12))
    lag_features = [f'lag_{i}' for i in range(1, 13)]

    for _ in range(12):
        last_values_df = pd.DataFrame([last_values_list[-12:]], columns=lag_features)
        new_pred = model_rf.predict(last_values_df.values)[0]  # Ensure values only are passed
        forecast_rf.append(new_pred)
        last_values_list.append(new_pred)

    return forecast_rf


def estimate_price(date_str, forecast, forecast_dates):
    """
    Estimate the oil price for a given date using forecasted values.

    Parameters:
    :param date_str: str - Date in the format "YYYY-MM-DD"
    :param forecast: np.array - Forecasted values
    :param forecast_dates: pd.DatetimeIndex - Dates for forecasted values

    Returns:
    :return: float/str - Estimated oil price or a message if the date is out of range
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    if date_obj in data['Dates'].values:
        oil_price = data.loc[data['Dates'] == date_obj, 'Prices'].values[0]
        return oil_price
    elif date_obj in forecast_dates:
        oil_price = forecast[forecast_dates.get_loc(date_obj)]
        return oil_price
    else:
        return "Date is out of range."


if __name__ == '__main__':

    ####################################################
    #       Model-1: Exponential Smoothing model
    ####################################################

    # Input full path of CSV file
    data_path = r"C:\Users\adars\Downloads\Laptop\Drive\Programming_resources\Python\JPM Online Internship\Task-1\Nat_Gas.csv"
    
    data = load_data(data_path)
    forecast, forecast_dates = exponential_smoothing_forecast(data)
    # Close the plot to proceed to the rest of the code

    # Testing the code for Model-1
    print(estimate_price("2024-09-30", forecast, forecast_dates))
    print(estimate_price("2022-06-30", forecast, forecast_dates))

    # Uncomment the below code and run to execute the other two models - ARIMA and Random Forest

    # ####################################################
    # #                Model-2: ARIMA
    # ####################################################
    # forecast_arima = arima_forecast(data)

    # ####################################################
    # #           Model-3: Random Forest Model
    # ####################################################
    # forecast_rf = random_forest_forecast(data)
    

