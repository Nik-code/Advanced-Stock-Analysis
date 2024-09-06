import pandas as pd
import numpy as np
from lstm_model import LSTMStockPredictor
from arima_model import ARIMAStockPredictor
import sys
import os
import json
import csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.data_collection import fetch_historical_data
import asyncio
import logging
from sklearn.model_selection import train_test_split
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import traceback
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROGRESS_FILE = os.path.join(os.path.dirname(__file__), 'training_progress.json')
FAILED_STOCKS_FILE = os.path.join(os.path.dirname(__file__), 'failed_stocks.csv')

MAX_CONSECUTIVE_ERRORS = 10
PAUSE_DURATION = 60  # 1 minute

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_data_for_stock(stock_code, time_frame='5years'):
    time.sleep(0.7) # 0.7 seconds delay to avoid rate limiting
    try:
        data = await fetch_historical_data(stock_code, time_frame)
        if data is not None and not data.empty:
            return data
        else:
            logger.warning(f"No data fetched for {stock_code}")
            save_failed_stock(stock_code, "No data fetched")
    except Exception as e:
        logger.error(f"Error fetching data for {stock_code}: {str(e)}")
        save_failed_stock(stock_code, f"Error fetching data: {str(e)}")
        raise  # Re-raise the exception to trigger a retry
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
        logger.error(traceback.format_exc())
        return stock_code, None, None


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'last_index': 0, 'errors': []}


def save_progress(completed, last_index, errors):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'completed': list(completed), 'last_index': last_index, 'errors': errors}, f)


def save_failed_stock(stock_code, reason):
    with open(FAILED_STOCKS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([stock_code, reason, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


async def train_models(stock_codes):
    progress = load_progress()
    completed = set(progress['completed'])
    start_index = progress['last_index']
    errors = progress['errors']
    
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
                errors.append(f"Missing data for {stock_code}")
                save_failed_stock(stock_code, "Missing data")
            
            # Save progress after each stock is processed (whether successful or not)
            completed.add(stock_code)
            save_progress(completed, i + 1, errors)
            logger.info(f"Progress: {len(completed)}/{total_stocks} stocks processed")
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed training for {result[0]}")
                # Update progress after each model is trained
                completed.add(result[0])
                save_progress(completed, start_index + len(completed), errors)
            except Exception as e:
                logger.error(f"Error processing future: {str(e)}")
                logger.error(traceback.format_exc())
                errors.append(f"Error processing {result[0] if result else 'unknown stock'}: {str(e)}")
            
            # Update progress after each future is completed
            save_progress(completed, start_index + len(completed), errors)
    
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
    try:
        csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'indian_stocks.csv')
        stocks_df = pd.read_csv(csv_path)
        stock_codes = stocks_df['tradingsymbol'].tolist()

        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        
        # Uncomment the following line if you want to delete existing models before training
        # delete_existing_models(model_dir)

        progress = load_progress()
        start_index = progress['last_index']
        completed = set(progress['completed'])

        stocks_to_train = [
            stock_code for stock_code in stock_codes[start_index:]
            if stock_code not in completed and should_retrain(os.path.join(model_dir, f'{stock_code}_lstm_model.h5'))
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
            logger.info("Training process completed successfully. Progress file removed.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())