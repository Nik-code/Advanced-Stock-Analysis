import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from app.models.lstm_model import LSTMStockPredictor
from app.models.arima_model import ARIMAStockPredictor
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backtest_lstm_model(stock_code, historical_data, investment_amount=10000, threshold=0.01, transaction_cost=0.001):
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    model_path = os.path.join(model_dir, f'{stock_code}_lstm_model.h5')
    scaler_path = os.path.join(model_dir, f'{stock_code}_scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found for {stock_code}")

    predictor = LSTMStockPredictor(input_shape=(60, 1))
    predictor.load_model(model_path)
    scaler = joblib.load(scaler_path)

    close_prices = historical_data['close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    predictions = predictor.predict(X)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

    cash = investment_amount
    shares = 0
    trades = []
    portfolio_values = [investment_amount]

    for i in range(len(predictions)):
        current_price = actual_prices[i][0]
        predicted_price = predictions[i][0]

        if i > 0:  # Add a one-day delay to simulate real-world latency
            prev_prediction = predictions[i-1][0]
            prev_price = actual_prices[i-1][0]

            if prev_prediction > prev_price * (1 + threshold) and cash > current_price:
                shares_to_buy = (cash * (1 - transaction_cost)) // current_price
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    cash -= shares_to_buy * current_price * (1 + transaction_cost)
                    trades.append(('buy', shares_to_buy, current_price))
            elif prev_prediction < prev_price * (1 - threshold) and shares > 0:
                cash += shares * current_price * (1 - transaction_cost)
                trades.append(('sell', shares, current_price))
                shares = 0

        portfolio_value = cash + shares * current_price
        portfolio_values.append(portfolio_value)

    final_portfolio_value = cash + shares * actual_prices[-1][0]
    total_return = (final_portfolio_value - investment_amount) / investment_amount * 100
    
    return {
        'initial_investment': investment_amount,
        'final_portfolio_value': final_portfolio_value,
        'total_return_percentage': total_return,
        'number_of_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values
    }


def backtest_arima_model(stock_code, historical_data, investment_amount=10000, threshold=0.01, transaction_cost=0.001):
    try:
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        model_path = os.path.join(model_dir, f'{stock_code}_arima_model.pkl')

        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Model directory: {model_dir}")
        logger.debug(f"Model path: {model_path}")
        logger.debug(f"Model exists: {os.path.exists(model_path)}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found for {stock_code}")

        predictor = joblib.load(model_path)
        logger.debug(f"Predictor type: {type(predictor)}")

        if not isinstance(predictor, ARIMAStockPredictor):
            raise TypeError(f"Loaded model is not an instance of ARIMAStockPredictor")

        close_prices = historical_data['close'].values

        cash = investment_amount
        shares = 0
        trades = []
        portfolio_values = [investment_amount]

        for i in range(1, len(close_prices)):
            current_price = close_prices[i]
            predicted_price = predictor.predict(steps=1)[0]

            if predicted_price > current_price * (1 + threshold) and cash > current_price:
                shares_to_buy = int((cash * (1 - transaction_cost)) // current_price)
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    cash -= shares_to_buy * current_price * (1 + transaction_cost)
                    trades.append(('buy', shares_to_buy, current_price))
            elif predicted_price < current_price * (1 - threshold) and shares > 0:
                cash += shares * current_price * (1 - transaction_cost)
                trades.append(('sell', shares, current_price))
                shares = 0

            portfolio_value = cash + shares * current_price
            portfolio_values.append(portfolio_value)

            # Update the model with the actual price
            predictor.train(close_prices[:i+1])

        final_portfolio_value = cash + shares * close_prices[-1]
        total_return = (final_portfolio_value - investment_amount) / investment_amount * 100
        
        return {
            'initial_investment': investment_amount,
            'final_portfolio_value': final_portfolio_value,
            'total_return_percentage': total_return,
            'number_of_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values
        }
    except Exception as e:
        logger.error(f"Error in backtest_arima_model: {str(e)}")
        raise