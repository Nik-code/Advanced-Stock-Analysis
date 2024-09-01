import pandas as pd
import numpy as np
from lstm_model import LSTMStockPredictor
import sys
import os
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

async def get_data_for_stock(stock_code, time_frame='5years'):
    try:
        data = await fetch_historical_data(stock_code, time_frame)
        if data is not None and not data.empty:
            return data
        else:
            logger.warning(f"No data fetched for {stock_code}")
    except Exception as e:
        logger.error(f"Error fetching data for {stock_code}: {str(e)}")
    return None

def train_and_save_model_for_stock(stock_code, data):
    logger.info(f"Training model for {stock_code}")
    close_prices = data['close'].values.reshape(-1, 1)

    predictor = LSTMStockPredictor(input_shape=(60, 1))
    X, y, scaler = predictor.preprocess_data(close_prices)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    history = predictor.train(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

    mae, rmse = predictor.evaluate(X_test, y_test)
    logger.info(f"Model Evaluation for {stock_code} - MAE: {mae}, RMSE: {rmse}")

    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f'{stock_code}_lstm_model.h5')
    scaler_path = os.path.join(model_dir, f'{stock_code}_scaler.pkl')
    
    predictor.save_model(model_path)
    joblib.dump(scaler, scaler_path)

    logger.info(f"Model and scaler for {stock_code} saved successfully.")
    return stock_code, mae, rmse

async def train_models(stock_codes):
    results = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for stock_code in stock_codes:
            data = await get_data_for_stock(stock_code)
            if data is not None:
                future = executor.submit(train_and_save_model_for_stock, stock_code, data)
                futures.append(future)
            else:
                logger.error(f"Skipping model training for {stock_code} due to missing data.")
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return results

def should_retrain(model_path, retrain_interval_days=30):
    if not os.path.exists(model_path):
        return True
    last_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
    return (datetime.now() - last_modified) > timedelta(days=retrain_interval_days)

async def main():
    stock_codes = [
        # Original list
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "WIPRO",
        
        # Additional stocks
        "BAJFINANCE", "HCLTECH", "ULTRACEMCO", "ADANIPORTS", "TITAN",
        "SUNPHARMA", "NESTLEIND", "BAJAJFINSV", "NTPC", "POWERGRID",
        "ONGC", "M&M", "GRASIM", "DRREDDY", "EICHERMOT",
        "COALINDIA", "TECHM", "INDUSINDBK", "CIPLA", "BPCL",
        "ADANIENT", "TATASTEEL", "JSWSTEEL", "HINDALCO", "TATAMOTORS",
        "BRITANNIA", "APOLLOHOSP", "HEROMOTOCO", "ADANIGREEN", "UPL",
        "DIVISLAB", "BAJAJ-AUTO", "SHREECEM", "ZOMATO", "PAYTM",
        "NYKAA", "POLICYBZR", "SIEMENS", "SAIL", "BIOCON",
        "JINDALSTEL", "PNB", "BANKBARODA", "DLF", "VEDL"
    ]

    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    stocks_to_train = [
        stock_code for stock_code in stock_codes
        if should_retrain(os.path.join(model_dir, f'{stock_code}_lstm_model.h5'))
    ]

    if stocks_to_train:
        results = await train_models(stocks_to_train)
        for stock_code, mae, rmse in results:
            logger.info(f"Training completed for {stock_code} - MAE: {mae}, RMSE: {rmse}")
    else:
        logger.info("No models need retraining at this time.")

if __name__ == "__main__":
    asyncio.run(main())