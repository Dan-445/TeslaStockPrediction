# Import necessary libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt  # For plotting
from prophet import Prophet
import joblib

# Step 1: Fetch Tesla Stock Price Data from Yahoo Finance
ticker = 'TSLA'
stock_data = yf.download(ticker, period='10y', interval='1d')

# Check if the data was downloaded successfully
if stock_data.empty:
    raise ValueError("No data was downloaded from Yahoo Finance. Please check the ticker symbol or try a different interval.")

# Display the first few rows of the data
print("Tesla Stock Price Data from Yahoo Finance (Daily):")
print(stock_data.head())

# Step 2: Prepare the dataframe for Prophet
# Prophet requires 'ds' (date) and 'y' (target variable) column names
stock_data.reset_index(inplace=True)
stock_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Verify the columns
print("Stock data columns after renaming:")
print(stock_data.columns)

# Check for missing values
print("Check for missing values:")
print(stock_data.isnull().sum())

# Step 3: Create Prophet Model with Regressors (Open, High, Low)
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.add_regressor('Open')
model.add_regressor('High')
model.add_regressor('Low')

# Step 4: Train the Model
# Ensure there are no NaN values in the columns used for regressors
stock_data[['Open', 'High', 'Low']] = stock_data[['Open', 'High', 'Low']].fillna(method='ffill')
stock_data[['Open', 'High', 'Low']] = stock_data[['Open', 'High', 'Low']].fillna(0)  # Fill remaining NaNs with 0 as a fallback

# Fit the model
model.fit(stock_data[['ds', 'y', 'Open', 'High', 'Low']])

# Save the trained model
model_filename = 'prophet_tesla_model_with_regressors.pkl'
joblib.dump(model, model_filename)
print(f"Trained model with regressors saved as {model_filename}")

# Step 5: Create Future Dates for Predictions
future_dates = model.make_future_dataframe(periods=7, freq='D')
future_dates = pd.merge(future_dates, stock_data[['ds', 'Open', 'High', 'Low']], on='ds', how='left')

# Handle any missing values in future dates (forward fill)
future_dates[['Open', 'High', 'Low']] = future_dates[['Open', 'High', 'Low']].fillna(method='ffill')
future_dates[['Open', 'High', 'Low']] = future_dates[['Open', 'High', 'Low']].fillna(0)

# Step 6: Predict Future Prices
forecast = model.predict(future_dates)

# Step 7: Save and Display the Predictions
forecast_filtered = forecast[forecast['ds'] >= pd.Timestamp.today().strftime('%Y-%m-%d')]
forecast_filtered.to_csv('tesla_predictions_with_open_high_low.csv', index=False)
print("Predictions saved to 'tesla_predictions_with_open_high_low.csv'")

# Display the Forecast
print(forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Open', 'High', 'Low']].head(10))

# Step 8: Visualize the Predictions with Confidence Intervals
def plot_with_confidence_intervals(forecast_df):
    """
    Plots the predictions along with confidence intervals.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the predicted values
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Predicted', color='blue', marker='o')
    
    # Plot the confidence intervals
    plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                     color='gray', alpha=0.3, label='Confidence Interval')
    
    # Formatting the plot
    plt.title('Predicted Tesla Stock Prices with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Save the plot as an image file
    plt.savefig('tesla_forecast_with_confidence_intervals.png')
    plt.show()

# Plot the forecast with confidence intervals
plot_with_confidence_intervals(forecast_filtered)
