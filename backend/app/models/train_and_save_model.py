import pandas as pd
import numpy as np
from lstm_model import LSTMStockPredictor
from app.services.data_collection import fetch_historical_data
import asyncio
import logging
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_data_for_stocks(stock_codes, time_frame='5years'):
    all_data = []
    for code in stock_codes:
        try:
            data = await fetch_historical_data(code, time_frame)
            if data is not None and not data.empty:
                all_data.append(data)
            else:
                logger.warning(f"No data fetched for {code}")
        except Exception as e:
            logger.error(f"Error fetching data for {code}: {str(e)}")
    return all_data

async def main():
    stock_codes = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "WIPRO"
    ]
    historical_data = await get_data_for_stocks(stock_codes)

    if not historical_data:
        logger.error("No data fetched. Exiting.")
        return

    combined_data = pd.concat(historical_data)
    combined_data = combined_data.sort_values(by='date')

    close_prices = combined_data['close'].values.reshape(-1, 1)

    predictor = LSTMStockPredictor(input_shape=(60, 1))
    X, y, scaler = predictor.preprocess_data(close_prices)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    history = predictor.train(X_train, y_train)

    mae, rmse = predictor.evaluate(X_test, y_test)
    logger.info(f"Model Evaluation - MAE: {mae}, RMSE: {rmse}")

    predictor.save_model('models/lstm_model.h5')
    joblib.dump(scaler, 'models/scaler.pkl')

    logger.info("Model and scaler saved successfully.")

if __name__ == "__main__":
    asyncio.run(main())