import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.models.XGBoost import XGBoostModel
from app.models.arima_model import ARIMAStockPredictor
from app.services.data_collection import fetch_historical_data
import asyncio
import logging
from app.models.random_forest import RandomForestModel

logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


async def train_and_save_models(stock_code):
    try:
        # Fetch 5 years of historical data
        historical_data = await fetch_historical_data(stock_code, '5years')
        if historical_data is None or len(historical_data) < 100:
            logger.error(f"Insufficient data for {stock_code}")
            return None

        close_prices = historical_data['close']

        # Prepare data
        X, y = prepare_data(close_prices.values)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'XGBoost': XGBoostModel(),
            'ARIMA': ARIMAStockPredictor(order=(1, 1, 1)),
            'RandomForest': RandomForestModel()
        }

        for name, model in models.items():
            try:
                if name == 'XGBoost':
                    model.train(close_prices.values)
                elif name == 'ARIMA':
                    model.train(close_prices.values.flatten())
                elif name == 'RandomForest':
                    model.train(X_train, y_train)

                # Save the model
                save_path = os.path.join(MODEL_SAVE_DIR, f'{stock_code}_{name}_model')
                model.save(save_path)
                logger.info(f"Saved {name} model for {stock_code}")
            except Exception as e:
                logger.error(f"Error training or saving {name} model for {stock_code}: {str(e)}")
                models[name] = None

        return models
    except Exception as e:
        logger.error(f"Error in train_and_save_models for {stock_code}: {str(e)}")
        return None


def prepare_data(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def backtest_model(model, X_test, y_test, historical_data=None):
    try:
        if isinstance(model, XGBoostModel):
            predictions = model.predict(X_test)
        elif isinstance(model, ARIMAStockPredictor):
            predictions = model.predict(steps=len(y_test))
        elif isinstance(model, RandomForestModel):
            predictions = model.predict(X_test)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        predictions = predictions[:len(y_test)]
        y_test = y_test[:len(predictions)]

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        return mse, mae
    except Exception as e:
        logger.error(f"Error in backtest_model: {str(e)}")
        return None, None


async def compare_models(stock_code):
    try:
        models = await train_and_save_models(stock_code)
        if models is None:
            logger.error(f"Failed to train models for {stock_code}")
            return None

        historical_data = await fetch_historical_data(stock_code, '5years')
        if historical_data is None:
            logger.error(f"Failed to fetch historical data for {stock_code}")
            return None

        close_prices = historical_data['close'].values
        X, y = prepare_data(close_prices)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}
        for name, model in models.items():
            if model is not None:
                mse, mae = backtest_model(model, X_test, y_test, historical_data)
                if mse is not None and mae is not None:
                    results[name] = {'MSE': mse, 'MAE': mae}
                else:
                    logger.warning(f"Backtesting failed for {name} model")

        if not results:
            logger.error(f"No valid results for any model for {stock_code}")
            return None

        # Sort models by MSE
        sorted_results = sorted(results.items(), key=lambda x: x[1]['MSE'])

        logger.info(f"Model comparison for {stock_code}:")
        for name, metrics in sorted_results:
            logger.info(f"{name}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")

        best_model = sorted_results[0][0]
        logger.info(f"Best performing model for {stock_code}: {best_model}")

        return results
    except Exception as e:
        logger.error(f"Error in compare_models for {stock_code}: {str(e)}")
        return None


if __name__ == "__main__":
    stock_code = "RELIANCE"  # Change this to the stock code you want to analyze
    asyncio.run(compare_models(stock_code))
