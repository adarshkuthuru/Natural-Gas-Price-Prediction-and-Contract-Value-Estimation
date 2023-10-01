# Natural Gas Price Prediction and Contract Estimation

Welcome to the **Natural Gas Price Prediction and Contract Estimation** repository. This repository houses scripts and algorithms designed to forecast natural gas prices based on historical data and estimate the value of contracts.

## Overview

1. **Predict_nat_gas_prices.py**: Contains forecasting algorithms for predicting natural gas prices.
2. **Estimate_contract_value.py**: Estimates the value of a gas storage contract.

### Data

The primary dataset for this analysis is `Nat_Gas.csv`. This CSV file contains:

- **Dates**: Date of the observation.
- **Prices**: Natural gas price on the given date.

### Forecasting Methods

The `Predict_nat_gas_prices.py` script incorporates three forecasting models:

1. **Exponential Smoothing**: This model predicts future values as a weighted average of past observations.
2. **ARIMA**: Utilizes lagged values and forecast errors to predict future values.
3. **Random Forest**: A machine learning approach, making predictions using decision trees.

### Contract Value Estimation

The `Estimate_contract_value.py` script calculates the value of a gas storage contract, considering factors such as storage costs, injection and withdrawal costs, and the predicted price of natural gas.

## Usage

### Predicting Prices

1. Load your data:

   ```python
   data = load_data("path_to_Nat_Gas.csv")
   ```

2. Use any of the forecasting models:

   - Exponential Smoothing:

     ```python
     forecast, forecast_dates = exponential_smoothing_forecast(data)
     ```

   - ARIMA:

     ```python
     forecast_arima = arima_forecast(data)
     ```

   - Random Forest:

     ```python
     forecast_rf = random_forest_forecast(data)
     ```

3. Estimate prices for specific dates:

   ```python
   print(estimate_price("2024-09-30", forecast, forecast_dates))
   ```

### Estimating Contract Value

1. Load your data:

   ```python
   prices, dates = load_data("path_to_Nat_Gas.csv")
   ```

2. Define contract parameters and compute its value:

   ```python
   value = contract_value(injection_dates, withdrawal_dates, inject_rate, max_volume, 
                          storage_cost_per_day, injection_cost, withdrawal_cost, 
                          interpolate_price, amplitude, shift, slope, intercept, start_date, prices, days_from_start)
   ```

## Examples

For a CSV file named `Nat_Gas.csv` with columns 'Dates' and 'Prices', to forecast prices using the Exponential Smoothing model:

```python
data = load_data("Nat_Gas.csv")
forecast, forecast_dates = exponential_smoothing_forecast(data)
print(estimate_price("2024-09-30", forecast, forecast_dates))
```

To estimate the value of a contract:

```python
prices, dates = load_data("Nat_Gas.csv")
value = contract_value(injection_dates, withdrawal_dates, ... [other parameters])
print(value)
```

## Concluding Notes

The models provided are versatile and can be modified or extended as needed. Remember always to validate the predictions using a holdout sample or cross-validation to ensure reliability.

Feedback and contributions are welcome. Please raise any issues or pull requests on this GitHub repository.
