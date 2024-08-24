import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, List


class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def prepare_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df['target'] = df['currentValue'].shift(-1)  # Next day's price as target
        df = df.dropna()

        features = ['currentValue', 'totalTradedQuantity', 'daily_return', 'volatility', 'rsi']
        return df[features], df['target']

    def train(self, data: List[Dict[str, Any]]):
        X, y = self.prepare_data(data)

        if len(X) < 2:
            print("Not enough data to train the model. Skipping training.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model trained. Test MSE: {mse}")

    def predict(self, data: Dict[str, Any]) -> float:
        if not self.is_trained:
            return data['currentValue']  # Return current value if model is not trained

        features = ['currentValue', 'totalTradedQuantity', 'daily_return', 'volatility', 'rsi']
        X = pd.DataFrame([data])[features]
        return self.model.predict(X)[0]