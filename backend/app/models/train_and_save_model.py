import pandas as pd
import numpy as np
from lstm_model import LSTMStockPredictor
from arima_model import ARIMAStockPredictor
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.data_collection import fetch_historical_data
import asyncio
import logging
from sklearn.model_selection import train_test_split
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROGRESS_FILE = 'training_progress.json'

async def get_data_for_stock(stock_code, time_frame='5years'):
    try:
        data = await fetch_historical_data(stock_code, time_frame)
        if data is not None and not data.empty:
            await asyncio.sleep(0.34)  # Sleep for 0.34 seconds to respect the rate limit
            return data
        else:
            logger.warning(f"No data fetched for {stock_code}")
    except Exception as e:
        logger.error(f"Error fetching data for {stock_code}: {str(e)}")
    await asyncio.sleep(0.34)  # Sleep even if there's an error to maintain the rate limit
    return None

def train_and_save_model_for_stock(stock_code, data):
    try:
        logger.info(f"Training models for {stock_code}")
        if data.empty:
            logger.warning(f"No data available for {stock_code}")
            return stock_code, None, None

        close_prices = data['close'].values.reshape(-1, 1)

        # Train LSTM model
        lstm_predictor = LSTMStockPredictor(input_shape=(60, 1))
        X, y, scaler = lstm_predictor.preprocess_data(close_prices)
        if len(X) < 100:  # Arbitrary minimum length for training
            logger.warning(f"Insufficient data for {stock_code} to train LSTM model")
            return stock_code, None, None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        history = lstm_predictor.train(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
        lstm_mae, lstm_rmse = lstm_predictor.evaluate(X_test, y_test)
        logger.info(f"LSTM Model Evaluation for {stock_code} - MAE: {lstm_mae}, RMSE: {lstm_rmse}")

        # Train ARIMA model
        arima_predictor = ARIMAStockPredictor()
        arima_predictor.train(close_prices.flatten())
        arima_mae, arima_rmse = arima_predictor.evaluate(close_prices[-len(X_test):].flatten())
        logger.info(f"ARIMA Model Evaluation for {stock_code} - MAE: {arima_mae}, RMSE: {arima_rmse}")

        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        lstm_model_path = os.path.join(model_dir, f'{stock_code}_lstm_model.h5')
        arima_model_path = os.path.join(model_dir, f'{stock_code}_arima_model.pkl')
        scaler_path = os.path.join(model_dir, f'{stock_code}_scaler.pkl')
        
        lstm_predictor.save_model(lstm_model_path)
        joblib.dump(arima_predictor, arima_model_path)
        joblib.dump(scaler, scaler_path)

        logger.info(f"Models and scaler for {stock_code} saved successfully.")
        return stock_code, (lstm_mae, lstm_rmse), (arima_mae, arima_rmse)
    except Exception as e:
        logger.error(f"Error training models for {stock_code}: {str(e)}")
        return stock_code, None, None

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'last_index': 0}

def save_progress(completed, last_index):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'completed': completed, 'last_index': last_index}, f)

async def train_models(stock_codes):
    progress = load_progress()
    completed = set(progress['completed'])
    start_index = progress['last_index']
    
    results = []
    total_stocks = len(stock_codes)
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, stock_code in enumerate(stock_codes[start_index:], start=start_index):
            if stock_code in completed:
                continue
            data = await get_data_for_stock(stock_code)
            if data is not None:
                future = executor.submit(train_and_save_model_for_stock, stock_code, data)
                futures.append(future)
            else:
                logger.error(f"Skipping model training for {stock_code} due to missing data.")
            
            # Save progress after each stock is processed (whether successful or not)
            completed.add(stock_code)
            save_progress(list(completed), i + 1)
            logger.info(f"Progress: {len(completed)}/{total_stocks} stocks processed")
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing future: {str(e)}")
    
    return results

def should_retrain(model_path, retrain_interval_days=30):
    if not os.path.exists(model_path):
        return True
    last_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
    return (datetime.now() - last_modified) > timedelta(days=retrain_interval_days)

def delete_existing_models(model_dir):
    for file in os.listdir(model_dir):
        if file.endswith(('.h5', '.pkl')):
            os.remove(os.path.join(model_dir, file))
    logger.info("Deleted existing models")

async def main():
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'indian_stocks.csv')
    stocks_df = pd.read_csv(csv_path)
    stock_codes = stocks_df['tradingsymbol'].tolist()

    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    
    # Uncomment the following line if you want to delete existing models before training
    # delete_existing_models(model_dir)

    stocks_to_train = [
        stock_code for stock_code in stock_codes
        if should_retrain(os.path.join(model_dir, f'{stock_code}_lstm_model.h5'))
    ]

    if stocks_to_train:
        results = await train_models(stocks_to_train)
        for stock_code, lstm_metrics, arima_metrics in results:
            if lstm_metrics and arima_metrics:
                logger.info(f"Training completed for {stock_code} - LSTM: MAE: {lstm_metrics[0]}, RMSE: {lstm_metrics[1]} - ARIMA: MAE: {arima_metrics[0]}, RMSE: {arima_metrics[1]}")
            else:
                logger.warning(f"Training failed for {stock_code}")
    else:
        logger.info("No models need retraining at this time.")

    # Clean up progress file after successful completion
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

if __name__ == "__main__":
    asyncio.run(main())