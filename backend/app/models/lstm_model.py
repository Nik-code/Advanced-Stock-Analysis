from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib


class LSTMModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_model(self):
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(units=64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        self.model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        predictions = self.model.predict(X_scaled)
        return self.scaler.inverse_transform(predictions)

    def save(self, path):
        self.model.save(f"{path}.keras")
        joblib.dump(self.scaler, f"{path}_scaler.joblib")

    @classmethod
    def load(cls, path):
        model = cls((60, 1))
        model.model = load_model(f"{path}.keras")
        model.scaler = joblib.load(f"{path}_scaler.joblib")
        return model
