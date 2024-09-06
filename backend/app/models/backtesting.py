import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import joblib
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ARIMAStockPredictor:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None

    def train(self, data):
        self.model = ARIMA(data, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, steps):
        forecast = self.model_fit.forecast(steps)
        return np.array(forecast)

    def evaluate(self, y_test, predictions):
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return mae, rmse

class LSTMStockPredictor:
    def __init__(self, input_shape=(60, 1)):
        self.model = None
        self.input_shape = input_shape

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return mae, rmse

def backtest_model(stock_code, historical_data, model_type, investment_amount=10000, threshold=0.01, transaction_cost=0.001):
    try:
        close_prices = historical_data['close'].values

        if model_type == 'LSTM':
            model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
            model_path = os.path.join(model_dir, f'{stock_code}_lstm_model.h5')
            scaler_path = os.path.join(model_dir, f'{stock_code}_scaler.pkl')

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Model or scaler not found for {stock_code}")

            predictor = LSTMStockPredictor()
            predictor.load_model(model_path)
            scaler = joblib.load(scaler_path)

            scaled_data = scaler.transform(close_prices.reshape(-1, 1))
            X, y = [], []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)

            predictions = predictor.predict(X)
            predictions = scaler.inverse_transform(predictions)
            actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

        elif model_type == 'ARIMA':
            train_size = int(len(close_prices) * 0.8)
            train, test = close_prices[:train_size], close_prices[train_size:]

            predictor = ARIMAStockPredictor()
            predictor.train(train)

            predictions = predictor.predict(len(test))
            if len(predictions) != len(test):
                raise ValueError(f"ARIMA predictions length ({len(predictions)}) does not match test data length ({len(test)})")
            actual_prices = test.reshape(-1, 1)
            predictions = predictions.reshape(-1, 1)

            mae, rmse = predictor.evaluate(actual_prices, predictions)

        else:
            raise ValueError(f"Invalid model type: {model_type}")

        cash = investment_amount
        shares = 0
        trades = []
        portfolio_values = [investment_amount]

        for i in range(1, len(actual_prices)):
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
        
        mae, rmse = predictor.evaluate(X if model_type == 'LSTM' else actual_prices.flatten(), 
                                       y if model_type == 'LSTM' else predictions)

        return {
            'stock_code': stock_code,
            'model_type': model_type,
            'initial_investment': investment_amount,
            'final_portfolio_value': final_portfolio_value,
            'total_return_percentage': total_return,
            'number_of_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions.flatten().tolist(),
            'actual_values': actual_prices.flatten().tolist()
        }

    except Exception as e:
        logger.error(f"Error in backtesting {model_type} model for {stock_code}: {str(e)}")
        raise

def backtest_lstm_model(stock_code, historical_data, investment_amount=10000, threshold=0.01, transaction_cost=0.001):
    return backtest_model(stock_code, historical_data, 'LSTM', investment_amount, threshold, transaction_cost)

def backtest_arima_model(stock_code, historical_data, investment_amount=10000, threshold=0.01, transaction_cost=0.001):
    return backtest_model(stock_code, historical_data, 'ARIMA', investment_amount, threshold, transaction_cost)
