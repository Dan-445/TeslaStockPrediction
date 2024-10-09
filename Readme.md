
# Tesla Stock Price Forecasting Using Prophet

This project demonstrates how to use the `Prophet` model with Yahoo Finance stock data for forecasting Tesla's stock prices. The model incorporates additional regressors such as `Open`, `High`, and `Low` prices to improve prediction accuracy. The predictions are saved along with visualizations, which include confidence intervals.

## Prerequisites

Before you begin, make sure you have the following libraries installed:

- `yfinance`
- `prophet`
- `joblib`
- `pandas`
- `matplotlib`

If these libraries are not already installed, you can install them using the following commands:

```bash
pip install yfinance prophet joblib pandas matplotlib
```

## Project Structure

The project includes the following main scripts:

1. `train_script.py` - Script for training the model using Tesla's stock price data and incorporating additional regressors.
2. `predict_script.py` - Script for loading the trained model, making predictions, and generating visualizations for different time intervals.
3. `README.md` - Documentation on how to use the scripts.

## How to Run

### 1. Train the Model

1. **Open `train_script.py`** and run the script to train the model with Tesla's historical stock data.
2. The script will:
   - Download Tesla's stock price data from Yahoo Finance.
   - Prepare the data for the Prophet model by renaming columns to `ds` and `y`.
   - Train the Prophet model with additional regressors (`Open`, `High`, `Low`).
   - Save the trained model as `prophet_tesla_model_with_regressors.pkl`.

### 2. Make Predictions

1. **Open `predict_script.py`** and run the script to generate future stock price predictions.
2. The script will:
   - Load the pretrained model (`prophet_tesla_model_with_regressors.pkl`).
   - Fetch historical stock data for the same ticker (`TSLA`) and period.
   - Prepare future dates, excluding weekends, for the desired intervals (5-minute, 15-minute, 30-minute, hourly, and daily).
   - Make predictions using the Prophet model and save the results in CSV files:
     - `tesla_5-minute_predictions.csv`
     - `tesla_15-minute_predictions.csv`
     - `tesla_30-minute_predictions.csv`
     - `tesla_hourly_predictions.csv`
     - `tesla_daily_predictions.csv`
   - Create and save plots of the predictions with confidence intervals for each interval.

### 3. View Results

The prediction results are saved as CSV files and plotted as images:

- **CSV Files**: Contain the predicted `yhat` values, along with `yhat_lower` and `yhat_upper` confidence intervals for different time intervals.
- **Plots**: Visual representations of the predictions with shaded confidence intervals.

### 4. Logging

Both scripts utilize logging for better visibility. Check the console for logs on each step, errors, and status updates.

## Important Notes

- Make sure the trained model (`prophet_tesla_model_with_regressors.pkl`) is in the same directory as `predict_script.py` before running the prediction script.
- If the Yahoo Finance data for Tesla is unavailable or if the ticker symbol changes, update the `ticker` variable in both scripts.

## License

This project is for educational purposes only. Feel free to modify and use it as per your needs.

## Acknowledgments

This project utilizes the `Prophet` library developed by Facebook for time series forecasting. For more information, visit the [Prophet GitHub Repository](https://github.com/facebook/prophet).

---

Enjoy forecasting!
