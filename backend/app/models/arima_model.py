import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ARIMAStockPredictor:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None

    def train(self, data):
        self.model = ARIMA(data, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, steps):
        return self.model_fit.forecast(steps)

    def evaluate(self, y_test):
        predictions = self.predict(len(y_test))
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return mae, rmse
