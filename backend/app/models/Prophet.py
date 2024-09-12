from prophet import Prophet
import pandas as pd
import joblib

class ProphetModel:
    def __init__(self):
        self.model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                             changepoint_prior_scale=0.05, seasonality_prior_scale=10)

    def train(self, data):
        df = pd.DataFrame({'ds': data.index, 'y': data.values})
        self.model.fit(df)

    def predict(self, steps):
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def save(self, path):
        with open(f"{path}.json", 'w') as f:
            f.write(self.model.to_json())

    @classmethod
    def load(cls, path):
        model = cls()
        with open(f"{path}.json", 'r') as f:
            model.model = Prophet.from_json(f.read())
        return model
