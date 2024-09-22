import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error


class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        self.scaler = MinMaxScaler()
        self.mse = None
        self.mae = None

    def prepare_data(self, data, lookback=60):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def train(self, data):
        X, y = self.prepare_data(data)
        X_train, X_val = X[:-100], X[-100:]
        y_train, y_val = y[:-100], y[-100:]
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = self.model.predict(X_val)
        self.mse = mean_squared_error(y_val, y_pred)
        self.mae = mean_absolute_error(y_val, y_pred)

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
        joblib.dump((self.model, self.scaler, self.mse, self.mae), path)

    @classmethod
    def load(cls, path):
        model = cls()
        model.model, model.scaler, model.mse, model.mae = joblib.load(path)
        return model
