import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.models.lstm_model import LSTMModel
from app.models.XGBoost import XGBoostModel
from app.models.GRU import GRUModel
from app.models.arima_model import ARIMAStockPredictor
from app.services.data_collection import fetch_historical_data
import asyncio
import logging
from app.models.Prophet import ProphetModel

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

        close_prices = historical_data['close'].values

        # Prepare data
        X, y = prepare_data(close_prices)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'LSTM': LSTMModel(input_shape=(60, 1)),
            'XGBoost': XGBoostModel(),
            'GRU': GRUModel(input_shape=(60, 1)),
            'ARIMA': ARIMAStockPredictor(),
            'Prophet': ProphetModel()
        }

        for name, model in models.items():
            try:
                if name in ['LSTM', 'GRU']:
                    model.train(X_train, y_train)
                elif name == 'XGBoost':
                    model.train(close_prices)
                elif name == 'ARIMA':
                    model.train(close_prices.flatten())
                elif name == 'Prophet':
                    model.train(pd.Series(close_prices, index=historical_data.index))

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
    if isinstance(model, (LSTMModel, GRUModel)):
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        predictions = model.predict(X_test_reshaped).flatten()
    elif isinstance(model, XGBoostModel):
        predictions = model.predict(X_test)
    elif isinstance(model, ARIMAStockPredictor):
        predictions = model.predict(steps=len(y_test))
    elif isinstance(model, ProphetModel):
        future_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1), periods=len(y_test))
        forecast = model.predict(steps=len(y_test))
        predictions = forecast['yhat'].values
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    predictions = predictions[:len(y_test)]
    y_test = y_test[:len(predictions)]

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return mse, mae


async def compare_models(stock_code):
    models = await train_and_save_models(stock_code)
    if models is None:
        return None

    historical_data = await fetch_historical_data(stock_code, '5years')
    close_prices = historical_data['close'].values
    X, y = prepare_data(close_prices)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, model in models.items():
        if model is not None:
            mse, mae = backtest_model(model, X_test, y_test, historical_data)
            results[name] = {'MSE': mse, 'MAE': mae}

    # Sort models by MSE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['MSE'])

    print(f"Model comparison for {stock_code}:")
    for name, metrics in sorted_results:
        print(f"{name}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}")

    best_model = sorted_results[0][0]
    print(f"Best performing model for {stock_code}: {best_model}")

    return results


if __name__ == "__main__":
    stock_code = "RELIANCE"  # Change this to the stock code you want to analyze
    asyncio.run(compare_models(stock_code))
