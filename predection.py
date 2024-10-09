# Import necessary libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from prophet import Prophet
import logging
from datetime import datetime

# Setup logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to Load the Pretrained Model
def load_model(model_path):
    """
    Load the pretrained Prophet model.
    Args:
        model_path (str): Path to the saved Prophet model file.
    Returns:
        Prophet: Loaded Prophet model object.
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Model successfully loaded from {model_path}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise

# Function to Fetch Historical Stock Data from Yahoo Finance
def fetch_stock_data(ticker, period='10y', interval='1d'):
    """
    Fetch historical stock price data for the specified ticker.
    Args:
        ticker (str): Stock ticker symbol (e.g., 'TSLA' for Tesla).
        period (str): Period of data to download (default is '10y').
        interval (str): Data interval (default is '1d' for daily).
    Returns:
        pd.DataFrame: Stock price data with necessary columns.
    """
    stock_data = yf.download(ticker, period=period, interval=interval)
    
    # Check if data is available
    if stock_data.empty:
        logger.error(f"No data found for ticker {ticker}. Please check the symbol or interval.")
        raise ValueError("No data was downloaded from Yahoo Finance. Please check the ticker symbol or try a different interval.")
    
    # Prepare data for Prophet
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)  # Remove timezone information
    stock_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    logger.info(f"Stock data successfully fetched and prepared for {ticker}.")
    
    return stock_data

# Function to Prepare Future Data with Regressors and Exclude Weekends
def prepare_future_dates(model, stock_data, periods, freq):
    """
    Prepare future dates with regressors for prediction, excluding weekends.
    Args:
        model (Prophet): Trained Prophet model.
        stock_data (pd.DataFrame): Historical stock data with regressors.
        periods (int): Number of future periods to generate.
        freq (str): Frequency of future periods (e.g., '5min', '15min').
    Returns:
        pd.DataFrame: Future dates dataframe with regressors.
    """
    # Generate future dates using Prophet
    future_dates = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Exclude weekends from the future dates
    future_dates = future_dates[future_dates['ds'].dt.weekday < 5]  # Keep only Monday to Friday
    
    # Merge with stock data to include regressors
    future_dates = pd.merge(future_dates, stock_data[['ds', 'Open', 'High', 'Low']], on='ds', how='left')
    future_dates[['Open', 'High', 'Low']] = future_dates[['Open', 'High', 'Low']].fillna(method='ffill').fillna(0)
    logger.info(f"Future dates for frequency '{freq}' prepared with regressors, excluding weekends.")
    
    return future_dates

# Function to Predict Future Prices
def predict_future_prices(model, future_dates):
    """
    Use the Prophet model to predict future stock prices.
    Args:
        model (Prophet): Trained Prophet model.
        future_dates (pd.DataFrame): Future dates with regressors.
    Returns:
        pd.DataFrame: Predicted stock prices with confidence intervals.
    """
    forecast = model.predict(future_dates)
    logger.info("Future prices predicted successfully.")
    return forecast

# Function to Save Predictions and Plots
def save_predictions_and_plots(forecast, interval_name):
    """
    Save predictions to CSV and plot the forecast with confidence intervals.
    Args:
        forecast (pd.DataFrame): Forecasted values.
        interval_name (str): Interval name for labeling (e.g., '5-Minute').
    """
    # Save forecast to CSV
    csv_filename = f'tesla_{interval_name.lower()}_predictions.csv'
    forecast.to_csv(csv_filename, index=False)
    logger.info(f"Predictions saved to {csv_filename}.")
    
    # Plot forecast with confidence intervals
    plt.figure(figsize=(12, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='blue', marker='o')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3, label='Confidence Interval')
    plt.title(f'{interval_name} Interval Predictions of Tesla Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Save plot as image
    plot_filename = f'tesla_{interval_name.lower()}_predictions_plot.png'
    plt.savefig(plot_filename)
    plt.close()
    logger.info(f"Plot saved as {plot_filename}.")

# Main function to execute the prediction process
def main():
    # Paths and ticker symbol
    model_path = 'prophet_tesla_model_with_regressors.pkl'
    ticker = 'TSLA'
    
    # Load the pretrained model
    model = load_model(model_path)
    
    # Fetch historical stock data for the required regressors
    stock_data = fetch_stock_data(ticker)
    
    # Prepare future dates and predict for different intervals
    intervals = {'5-Minute': (2016, '5min'), '15-Minute': (672, '15min'), '30-Minute': (336, '30min'), 'Hourly': (168, 'H'), 'Daily': (7, 'D')}
    
    for interval_name, (periods, freq) in intervals.items():
        future_dates = prepare_future_dates(model, stock_data, periods, freq)
        forecast = predict_future_prices(model, future_dates)
        
        # Filter predictions for the next 7 days
        forecast_filtered = forecast[forecast['ds'] >= datetime.today().strftime('%Y-%m-%d')]
        
        # Save predictions and plots for each interval
        save_predictions_and_plots(forecast_filtered, interval_name)

if __name__ == "__main__":
    main()