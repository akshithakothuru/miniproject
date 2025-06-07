import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os
import logging
import re
import time
import random
import requests
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from flask import Flask, send_file, request, jsonify
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Financial Modeling Prep (FMP) API configuration (replace with your API key)
FMP_API_KEY = "XbWh9lX8rLWu35i4cSOwPRwT8w49sO2h"  # Replace with your actual API key from https://financialmodelingprep.com/developer

# Define U.S. market holidays for 2025 (NASDAQ)
MARKET_HOLIDAYS_2025 = [
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving Day
    "2025-12-25",  # Christmas Day
]
MARKET_HOLIDAYS_2025 = [pd.to_datetime(h) for h in MARKET_HOLIDAYS_2025]

def is_trading_day(date):
    """Check if the given date is a trading day (not a weekend or holiday)."""
    date = pd.to_datetime(date)
    # Check for weekends
    if date.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False
    # Check for holidays
    if date in MARKET_HOLIDAYS_2025:
        return False
    return True

def get_last_trading_day(date):
    """Get the last trading day before the given date."""
    date = pd.to_datetime(date)
    last_day = date - timedelta(days=1)
    while not is_trading_day(last_day):
        last_day -= timedelta(days=1)
    return last_day

def get_next_trading_day(date):
    """Get the next trading day after the given date."""
    date = pd.to_datetime(date)
    next_day = date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day

def load_todays_sentiment(ticker, sentiment_file='C:\\Users\\akshi\\mps\\sentiment_analysis_results.csv'):
    try:
        df = pd.read_csv(sentiment_file)
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
        pattern = re.compile(f'\\b{ticker}\\b', re.IGNORECASE)
        df = df[df['headline'].str.contains(pattern, na=False)]
        latest_date = df['Date'].max()
        df = df[df['Date'] == latest_date]
        if df.empty:
            logging.warning(f"No sentiment data for {ticker} on {latest_date}. Using price-only mode.")
            return 0.0
        df['Weighted_Score'] = df['sentiment_score'] * df['confidence']
        sentiment_score = df['Weighted_Score'].sum() / df['confidence'].sum()
        logging.info(f"Loaded sentiment for {ticker} on {latest_date}: {sentiment_score:.4f}")
        return sentiment_score
    except Exception as e:
        logging.error(f"Error loading sentiment: {e}")
        return 0.0

def validate_price_range(ticker, df):
    price_ranges = {
        'AAPL': (150, 300),
        'MSFT': (300, 500),
        'AMZN': (100, 250),
        'GOOGL': (120, 200)
    }
    if ticker in price_ranges:
        min_price, max_price = price_ranges[ticker]
        close_prices = df['Close']
        if close_prices.min() < min_price or close_prices.max() > max_price:
            return False
    return True

def fetch_fmp_data(ticker, start_date, end_date, max_retries=3, initial_delay=5):
    # Convert start_date and end_date to Timestamps for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Fetching data for {ticker} from {start_date} to {end_date} using Financial Modeling Prep (Attempt {attempt + 1})")
            # Convert dates to YYYY-MM-DD format for FMP API request
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # FMP endpoint for historical daily adjusted stock data
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date_str}&to={end_date_str}&apikey={FMP_API_KEY}"
            response = requests.get(url)
            
            # Check for errors
            if response.status_code == 429:
                raise Exception("Rate limit exceeded. Too many requests (250 calls/day limit).")
            if response.status_code == 401:
                raise Exception("Invalid API key. Please check your FMP API key.")
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            data = response.json()
            
            # Check for successful response
            if "historical" not in data:
                raise ValueError(f"No historical data retrieved from FMP: {data.get('Error Message', 'Unknown error')}")
            
            # Extract daily adjusted close prices
            dates = []
            close_prices = []
            for entry in data["historical"]:
                date = pd.to_datetime(entry["date"])
                close_price = entry["adjClose"]  # Adjusted close price
                if start_date <= date <= end_date:
                    dates.append(date)
                    close_prices.append(float(close_price))
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': dates,
                'Close': close_prices
            })
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone info
            df = df.set_index('Date')
            df = df.sort_index(ascending=True)
            
            expected_days = (end_date - start_date).days * 0.7
            if len(df) < 80:
                raise ValueError(f"Insufficient data: {len(df)} days retrieved, expected ~{int(expected_days)}")
            logging.info(f"Retrieved {len(df)} days of data from {df.index[0]} to {df.index[-1]}, latest price: {df['Close'].iloc[-1]:.2f}")
            return df
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Rate limit hit. Retrying after {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
            logging.error(f"Error fetching FMP data: {e}")
            raise

def fetch_stock_data(ticker, start_date, end_date, local_file_template='C:\\Users\\akshi\\mps\\stock_data_{ticker}.csv'):
    local_file = local_file_template.format(ticker=ticker)
    reference_prices = {'AAPL': 211.25, 'GOOGL': 154.0, 'MSFT': 435.0}
    reference_price = reference_prices.get(ticker, None)
    
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Check local CSV first
    if os.path.exists(local_file):
        try:
            logging.info(f"Checking local CSV file: {local_file}")
            df = pd.read_csv(local_file)
            if 'Date' not in df.columns and 'date' in df.columns:
                df = df.rename(columns={'date': 'Date'})
            elif 'Date' not in df.columns:
                df.columns = ['Date'] + list(df.columns[1:])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df.sort_index(ascending=True)
            df = df[(df.index >= start_date) & (df.index <= end_date)][['Close']]
            
            # Check if CSV has enough data and is within price range
            if len(df) >= 200 and validate_price_range(ticker, df):
                csv_price = float(df['Close'].iloc[-1])
                latest_date = df.index[-1]
                # Validate against reference price
                if reference_price and abs(csv_price - reference_price) > 0.03 * reference_price:
                    raise ValueError(f"CSV price {csv_price:.2f} deviates >3% from reference {reference_price:.2f}")
                
                # Check if CSV is up-to-date (within 5 days)
                if (end_date - latest_date).days > 5:
                    raise ValueError(f"Local CSV data is too old: last date {latest_date}, expected up to {end_date}")
                
                # Attempt to fetch recent data to append
                try:
                    next_day = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    live_df = fetch_fmp_data(ticker, next_day, end_date.strftime('%Y-%m-%d'))
                    live_price = float(live_df['Close'].iloc[-1])
                    if abs(csv_price - live_price) > 0.05 * live_price:
                        logging.warning(f"Latest CSV price {csv_price:.2f} deviates from live {live_price:.2f}. Fetching full data.")
                        df = fetch_fmp_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    else:
                        # Append live data to CSV data
                        df = pd.concat([df, live_df])
                        logging.info(f"Using local CSV with appended live data: {len(df)} days from {df.index[0]} to {df.index[-1]}, latest price: {df['Close'].iloc[-1]:.2f}")
                        return df
                except Exception as e:
                    logging.warning(f"Live validation failed: {e}. Checking reference price.")
                    if reference_price and abs(csv_price - reference_price) <= 0.03 * reference_price:
                        logging.info(f"Using local CSV with {len(df)} days from {df.index[0]} to {df.index[-1]}, latest price: {csv_price:.2f}")
                        return df
                    raise ValueError(f"CSV validation failed and no valid live data: {e}")
            else:
                logging.info(f"Local CSV invalid: {len(df)} days or incorrect price range")
        except Exception as csv_e:
            logging.error(f"Error reading CSV: {csv_e}")

    # Fetch data using FMP if local CSV is invalid or missing
    df = fetch_fmp_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df.reset_index().to_csv(local_file, index=False)
    logging.info(f"Saved new data to {local_file}")
    return df

def plot_stock_prices(ticker, df, predicted_prices, pred_dates):
    try:
        # Ensure all dates are datetime64[ns] and timezone-naive
        start_date = pd.to_datetime('2025-05-01').tz_localize(None)
        end_date = pd.to_datetime(df.index[-1]).tz_localize(None)
        pred_dates = [pd.to_datetime(date).tz_localize(None) for date in pred_dates]
        
        # Filter historical data
        historical = df[(df.index >= start_date) & (df.index <= end_date)][['Close']].copy()
        historical = historical.resample('D').last().ffill()
        
        # Create date range for plotting, ensuring all dates are timezone-naive
        max_pred_date = max(pred_dates)
        all_dates = pd.date_range(start=start_date, end=max_pred_date, freq='D')
        all_dates = [pd.to_datetime(date).tz_localize(None) for date in all_dates]
        
        # Prepare actual prices
        actual_prices = [float(historical['Close'].loc[date]) if date in historical.index else np.nan for date in all_dates]
        
        # Prepare predicted prices
        predicted_prices_list = [np.nan] * len(all_dates)
        for i, pred_date in enumerate(pred_dates):
            # Find the index by comparing dates directly
            idx = None
            for j, date in enumerate(all_dates):
                if date.date() == pred_date.date():
                    idx = j
                    break
            if idx is not None:
                predicted_prices_list[idx] = float(predicted_prices[i])
                logging.info(f"Placed predicted price {predicted_prices[i]:.2f} at index {idx} for date {pred_date.strftime('%Y-%m-%d')}")
            else:
                logging.warning(f"Could not find matching date for {pred_date.strftime('%Y-%m-%d')} in all_dates")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(all_dates, actual_prices, 'b-', label='Actual Close', marker='o')
        plt.plot(all_dates, predicted_prices_list, 'r--', label='Predicted Close', marker='x')
        plt.title(f'{ticker} Stock Price: May 1, 2025 to {max_pred_date.strftime("%Y-%m-%d")}')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.legend()
        plt.xticks(all_dates, [date.strftime('%Y-%m-%d') for date in all_dates], rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_file = f'C:\\Users\\akshi\\mps\\stock_price_plot_{ticker}.png'
        plt.savefig(plot_file)
        plt.close()
        logging.info(f"Plot saved to {plot_file}")
        return plot_file
    except Exception as e:
        logging.error(f"Error plotting stock prices: {e}")
        return None

def create_sequences(data, seq_length):
    try:
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    except Exception as e:
        logging.error(f"Error creating sequences: {e}")
        raise

def mean_absolute_percentage_error(y_true, y_pred):
    try:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    except Exception as e:
        logging.error(f"Error calculating MAPE: {e}")
        raise

def sanitize_json(data):
    """Recursively replace NaN, Infinity, etc., with None to make JSON serializable."""
    if isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(item) for item in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
        return data
    return data

@app.route('/predict-price', methods=['GET'])
def predict_price():
    ticker = request.args.get('ticker', 'AAPL').upper()
    sentiment_score = float(request.args.get('sentiment_score', 0.0))
    
    # Define date range for fetching data
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')  # Today: 2025-05-18
    start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')  # One year ago: 2024-05-18
    
    try:
        # Fetch stock data
        logging.info("Fetching stock data...")
        df = fetch_stock_data(ticker, start_date, end_date)
        if df.empty:
            return jsonify({"error": "No data available for processing."}), 500
        
        # Handle null values
        logging.info("Checking for null values in data...")
        if df['Close'].isnull().any():
            logging.warning("Null values detected in Close prices. Filling with forward fill.")
            df['Close'] = df['Close'].ffill()
        
        # Use the provided sentiment score
        logging.info(f"Using provided sentiment score for {ticker}: {sentiment_score:.4f}")
        
        # Prepare data for LSTM
        logging.info("Preparing data for LSTM...")
        data = df['Close'].values.reshape(-1, 1)
        logging.info(f"Data range: {df.index[0]} to {df.index[-1]}, {len(df)} records")
        logging.info(f"Min Close: {data.min().item():.2f}, Max Close: {data.max().item():.2f}, Latest Close: {data[-1][0]:.2f}")
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Split data into training and testing
        logging.info("Splitting data into training and testing sets...")
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        seq_length = 10
        X_train, y_train = create_sequences(train_data, seq_length)
        X_test, y_test = create_sequences(test_data, seq_length)
        
        # Build and train the LSTM model
        logging.info("Building and training LSTM model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(seq_length, 1)),
            tf.keras.layers.LSTM(units=200, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=200, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=150),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.1, verbose=0)
        
        # Make predictions for the next two trading days
        logging.info("Predicting prices for the next two trading days...")
        last_sequence = scaled_data[-seq_length:].reshape((1, seq_length, 1))
        
        # Predict for the first trading day
        first_trading_pred_scaled = model.predict(last_sequence, batch_size=4, verbose=0)
        first_trading_pred = scaler.inverse_transform(first_trading_pred_scaled)
        first_trading_pred_value = float(first_trading_pred[0][0])
        
        # Prepare sequence for the second trading day's prediction
        new_sequence = np.append(scaled_data[-seq_length+1:], first_trading_pred_scaled).reshape((1, seq_length, 1))
        second_trading_pred_scaled = model.predict(new_sequence, batch_size=4, verbose=0)
        second_trading_pred = scaler.inverse_transform(second_trading_pred_scaled)
        second_trading_pred_value = float(second_trading_pred[0][0])
        
        # Apply sentiment adjustments
        if sentiment_score > 0.5:
            first_trading_pred_value += 3.0
            second_trading_pred_value += 3.0
            logging.info(f"Applied highly positive sentiment adjustment: +$3.00")
        elif sentiment_score > 0.0:
            first_trading_pred_value += 0.2
            second_trading_pred_value += 0.2
            logging.info(f"Applied moderately positive sentiment adjustment: +$0.20")
        elif sentiment_score < -0.5:
            first_trading_pred_value -= 3.0
            second_trading_pred_value -= 3.0
            logging.info(f"Applied highly negative sentiment adjustment: -$3.00")
        else:
            first_trading_pred_value -= 0.2
            second_trading_pred_value -= 0.2
            logging.info(f"Applied moderately negative sentiment adjustment: -$0.20")
        
        # Determine the next two trading days
        first_trading_day = get_next_trading_day(today)  # Will be May 19, 2025 (Monday)
        second_trading_day = get_next_trading_day(first_trading_day)  # Will be May 20, 2025 (Tuesday)
        
        logging.info(f"Next trading day 1 ({first_trading_day.strftime('%Y-%m-%d')}) predicted price: ${first_trading_pred_value:.2f}")
        logging.info(f"Next trading day 2 ({second_trading_day.strftime('%Y-%m-%d')}) predicted price: ${second_trading_pred_value:.2f}")
        
        # Calculate evaluation metrics
        logging.info("Calculating evaluation metrics...")
        test_predictions = model.predict(X_test, batch_size=4, verbose=0)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_actual = scaler.inverse_transform(y_test)
        mae = float(np.mean(np.abs(y_test_actual - test_predictions)))
        mape = float(mean_absolute_percentage_error(y_test_actual, test_predictions))
        
        logging.info(f"Test MAE: {mae:.2f}")
        logging.info(f"Test MAPE: {mape:.2f}%")
        
        # Save predictions to CSV
        logging.info("Saving predictions to CSV...")
        output_df = pd.DataFrame({
            'Ticker': [ticker, ticker],
            'Date': [first_trading_day, second_trading_day],
            'Predicted_Close': [first_trading_pred_value, second_trading_pred_value],
            'MAE': [mae, mae],
            'MAPE': [mape, mape]
        })
        output_file = 'C:\\Users\\akshi\\mps\\stock_predictions.csv'
        mode = 'a' if os.path.exists(output_file) else 'w'
        output_df.to_csv(output_file, mode=mode, header=mode=='w', index=False)
        logging.info(f"Predictions saved to {output_file}")
        
        # Generate plot
        logging.info("Generating stock price plot...")
        plot_file = plot_stock_prices(ticker, df, [first_trading_pred_value, second_trading_pred_value], [first_trading_day, second_trading_day])
        plot_url = f"/plots/stock_price_plot_{ticker}.png" if plot_file else None
        
        # Return response
        logging.info("Returning response...")
        response = {
            "ticker": ticker,
            "predictions": [
                {"date": first_trading_day.strftime('%Y-%m-%d'), "predicted_price": first_trading_pred_value},
                {"date": second_trading_day.strftime('%Y-%m-%d'), "predicted_price": second_trading_pred_value}
            ],
            "mae": mae,
            "mape": mape,
            "plot_url": plot_url
        }
        sanitized_response = sanitize_json(response)
        logging.info(f"Returning sanitized response: {sanitized_response}")
        return jsonify(sanitized_response)
    except Exception as e:
        logging.error(f"Error in price prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_file(os.path.join('C:\\Users\\akshi\\mps', filename))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)