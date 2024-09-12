import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib


class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        self.scaler = MinMaxScaler()

    def prepare_data(self, data, lookback=60):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def train(self, data):
        X, y = self.prepare_data(data)
        self.model.fit(X, y)

    def predict(self, data, steps=7):
        X, _ = self.prepare_data(data)
        last_sequence = X[-1]
        predictions = []
        for _ in range(steps):
            prediction = self.model.predict(last_sequence.reshape(1, -1))
            predictions.append(prediction[0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = prediction[0]
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    def save(self, path):
        joblib.dump(self.model, f"{path}.joblib")

    @classmethod
    def load(cls, path):
        model = cls()
        model.model = joblib.load(f"{path}.joblib")
        return model
