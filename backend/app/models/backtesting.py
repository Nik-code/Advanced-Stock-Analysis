import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import joblib
import logging
import os
from pydantic import BaseModel
from app.models.backtesting import backtest_lstm_model, backtest_arima_model
from app.services.data_collection import fetch_historical_data

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


def backtest_model(stock_code, historical_data, model_type, investment_amount=10000, threshold=0.01, transaction_cost=0.001, stop_loss_percentage=0.05):
    try:
        close_prices = historical_data['close'].values
        cash = investment_amount
        shares = 0
        trades = []
        portfolio_values = [investment_amount]
        buy_price = 0
        mae = 0
        rmse = 0
        final_portfolio_value = investment_amount

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
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actual_prices = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

            logger.info(f"LSTM Predictions for {stock_code}:")
            for i in range(len(predictions)):
                current_price = actual_prices[i]
                current_prediction = predictions[i]

                if i > 0:
                    prev_price = actual_prices[i-1]

                    if shares > 0 and current_price < buy_price * (1 - stop_loss_percentage):
                        cash += shares * current_price * (1 - transaction_cost)
                        trades.append(('sell', shares, current_price, 'stop-loss'))
                        shares = 0
                    elif current_prediction > prev_price * (1 + threshold) and cash > current_price:
                        shares_to_buy = (cash * (1 - transaction_cost)) // current_price
                        if shares_to_buy > 0:
                            shares += shares_to_buy
                            cash -= shares_to_buy * current_price * (1 + transaction_cost)
                            trades.append(('buy', shares_to_buy, current_price))
                            buy_price = current_price
                    elif current_prediction < prev_price * (1 - threshold) and shares > 0:
                        cash += shares * current_price * (1 - transaction_cost)
                        trades.append(('sell', shares, current_price))
                        shares = 0

                portfolio_value = cash + shares * current_price
                portfolio_values.append(portfolio_value)

            final_portfolio_value = cash + shares * actual_prices[-1]

        elif model_type == 'ARIMA':
            # ARIMA model implementation
            model = ARIMA(close_prices, order=(1, 1, 1))
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(close_prices))
            actual_prices = close_prices

            mae = np.mean(np.abs(predictions - actual_prices))
            rmse = np.sqrt(np.mean((predictions - actual_prices)**2))

            for i in range(1, len(predictions)):
                current_price = actual_prices[i]
                prev_price = actual_prices[i-1]
                prev_prediction = predictions[i-1]

                if shares > 0 and current_price < buy_price * (1 - stop_loss_percentage):
                    cash += shares * current_price * (1 - transaction_cost)
                    trades.append(('sell', shares, current_price, 'stop-loss'))
                    shares = 0
                elif prev_prediction > prev_price * (1 + threshold) and cash > current_price:
                    shares_to_buy = (cash * (1 - transaction_cost)) // current_price
                    if shares_to_buy > 0:
                        shares += shares_to_buy
                        cash -= shares_to_buy * current_price * (1 + transaction_cost)
                        trades.append(('buy', shares_to_buy, current_price))
                        buy_price = current_price
                elif prev_prediction < prev_price * (1 - threshold) and shares > 0:
                    cash += shares * current_price * (1 - transaction_cost)
                    trades.append(('sell', shares, current_price))
                    shares = 0

                portfolio_value = cash + shares * current_price
                portfolio_values.append(portfolio_value)

            final_portfolio_value = cash + shares * actual_prices[-1]

        else:
            raise ValueError(f"Invalid model type: {model_type}")

        total_return = (final_portfolio_value - investment_amount) / investment_amount * 100

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
            'predictions': predictions.tolist(),
            'actual_values': actual_prices.tolist()
        }

    except Exception as e:
        logger.error(f"Error in backtesting {model_type} model for {stock_code}: {str(e)}")
        raise


def backtest_lstm_model(stock_code, historical_data, investment_amount=10000, threshold=0.01, transaction_cost=0.001):
    return backtest_model(stock_code, historical_data, 'LSTM', investment_amount, threshold, transaction_cost)


def backtest_arima_model(stock_code, historical_data, investment_amount=10000, threshold=0.01, transaction_cost=0.001):
    return backtest_model(stock_code, historical_data, 'ARIMA', investment_amount, threshold, transaction_cost)
