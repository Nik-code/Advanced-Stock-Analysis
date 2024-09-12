from prophet import Prophet
import pandas as pd


class ProphetModel:
    def __init__(self):
        self.model = Prophet(daily_seasonality=True)

    def prepare_data(self, data):
        df = pd.DataFrame({'ds': data.index, 'y': data.values})
        return df

    def train(self, data):
        df = self.prepare_data(data)
        self.model.fit(df)

    def predict(self, steps=7):
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
