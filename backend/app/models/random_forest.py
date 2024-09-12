from sklearn.ensemble import RandomForestRegressor
import joblib


class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, f"{path}.joblib")

    @classmethod
    def load(cls, path):
        model = cls()
        model.model = joblib.load(f"{path}.joblib")
        return model
