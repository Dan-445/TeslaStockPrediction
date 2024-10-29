import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import joblib

# Step 1: Fetch Tesla Stock Price Data from 2010-01-01 to 2024-10-04
ticker = 'TSLA'
tesla_data = yf.download(ticker, start='2010-01-01', end='2024-10-04', interval='1d')

# Check if data was downloaded successfully
if tesla_data.empty:
    raise ValueError("No data was downloaded from Yahoo Finance. Please check the ticker symbol or try a different interval.")

# Prepare the dataframe for Prophet with additional regressors
tesla_data.reset_index(inplace=True)
tesla_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Fine-tune the Prophet model parameters
model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.1,   # Increased flexibility in trend changepoints
    seasonality_prior_scale=0.5    # Increased seasonality control
)
model.add_regressor('Open')
model.add_regressor('High')
model.add_regressor('Low')
model.add_regressor('Volume')

# Step 2: Prepare and Train the Model with Regressors
tesla_data[['Open', 'High', 'Low', 'Volume']] = tesla_data[['Open', 'High', 'Low', 'Volume']].ffill()
model.fit(tesla_data[['ds', 'y', 'Open', 'High', 'Low', 'Volume']])

# Step 3: Save the Trained Model
model_filename = 'prophet_tesla_model_finetuned.pkl'
joblib.dump(model, model_filename)
print(f"Trained model saved as '{model_filename}'")

# Step 4: Create Future Dates for Predictions
future_dates = tesla_data[['ds', 'Open', 'High', 'Low', 'Volume']]

# Step 5: Predict Prices
forecast = model.predict(future_dates)

# Step 6: Merge Actual and Predicted Values
comparison_df = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], tesla_data[['ds', 'y']], on='ds', how='inner')
comparison_df.rename(columns={'y': 'Actual', 'yhat': 'Predicted'}, inplace=True)

# Step 7: Calculate Accuracy Metrics (RMSE, MAE, MAPE)
rmse = np.sqrt(mean_squared_error(comparison_df['Actual'], comparison_df['Predicted']))
mae = mean_absolute_error(comparison_df['Actual'], comparison_df['Predicted'])
mape = mean_absolute_percentage_error(comparison_df['Actual'], comparison_df['Predicted']) * 100

print(f"Model Accuracy Metrics:\nRMSE: {rmse}\nMAE: {mae}\nMAPE: {mape}%")

# Step 8: Plot Actual vs Predicted Prices with Confidence Intervals
def plot_comparison_with_confidence_intervals(comparison_df, rmse, mape):
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['ds'], comparison_df['Actual'], label='Actual', color='red', marker='o', markersize=3)
    plt.plot(comparison_df['ds'], comparison_df['Predicted'], label='Predicted', color='blue', marker='o', markersize=3)
    plt.fill_between(comparison_df['ds'], comparison_df['yhat_lower'], comparison_df['yhat_upper'], color='blue', alpha=0.2)
    plt.title(f'Actual vs Predicted Tesla Stock Prices\nRMSE: {rmse:.2f}, MAPE: {mape:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Step 9: Plot the actual vs predicted prices with confidence intervals and metrics
plot_comparison_with_confidence_intervals(comparison_df, rmse, mape)
