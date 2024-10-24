import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.mse = None
        self.mae = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_train)
        self.mse = mean_squared_error(y_train, y_pred)
        self.mae = mean_absolute_error(y_train, y_pred)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
