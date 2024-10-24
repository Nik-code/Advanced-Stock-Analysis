import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import joblib


class ARIMAStockPredictor:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.mse = None
        self.mae = None

    def train(self, data):
        self.model = ARIMA(data, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, steps=1):
        if self.model_fit is None:
            raise ValueError("Model has not been trained yet.")
        return self.model_fit.forecast(steps=steps)

    def evaluate(self, test_data):
        predictions = self.predict(len(test_data))
        mse = np.mean((test_data - predictions) ** 2)
        rmse = np.sqrt(mse)
        return mse, rmse

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def calculate_metrics(self, test_data):
        predictions = self.predict(len(test_data))
        self.mse = np.mean((test_data - predictions) ** 2)
        self.mae = np.mean(np.abs(test_data - predictions))
