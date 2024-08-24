import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, List


class StockPredictor:
    def __init__(self, model_type: str = 'linear'):
        self.model_type = model_type
        self.model = self._initialize_model(model_type)
        self.is_trained = False
        self.scaler = StandardScaler()  # For scaling features

    def _initialize_model(self, model_type: str):
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported model type. Choose either 'linear' or 'random_forest'.")

    def prepare_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(data)

        # Add new features
        df['ma50'] = df['currentValue'].rolling(window=50).mean()
        df['ma200'] = df['currentValue'].rolling(window=200).mean()
        df['volatility'] = df['currentValue'].rolling(window=20).std()
        df['rsi'] = calculate_rsi(df['currentValue'])

        # Target variable is the next day's price
        df['target'] = df['currentValue'].shift(-1)
        df = df.dropna()

        features = ['currentValue', 'totalTradedQuantity', 'daily_return', 'volatility', 'rsi', 'ma50', 'ma200']
        return df[features], df['target']

    def train(self, data: List[Dict[str, Any]]):
        X, y = self.prepare_data(data)

        if len(X) < 2:
            print("Not enough data to train the model. Skipping training.")
            return

        # Scaling the features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model trained. Test MSE: {mse}")

    def predict(self, data: Dict[str, Any]) -> float:
        if not self.is_trained:
            return data['currentValue']  # Return current value if model is not trained

        features = ['currentValue', 'totalTradedQuantity', 'daily_return', 'volatility', 'rsi', 'ma50', 'ma200']
        X = pd.DataFrame([data])[features]

        # Scale the features
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
