import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib

class LSTMModel:
    def __init__(self, input_shape, sequence_length=60, units=50, dropout=0.2, learning_rate=0.001):
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_model(self):
        model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=self.input_shape),
            Dropout(self.dropout),
            LSTM(units=self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        self.model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        predictions = self.model.predict(X_scaled)
        return self.scaler.inverse_transform(predictions)

    def save(self, path):
        self.model.save(f"{path}_model.h5")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")

    @classmethod
    def load(cls, path):
        model = cls(input_shape=(60, 1))  # Adjust input_shape as needed
        model.model = load_model(f"{path}_model.h5")
        model.scaler = joblib.load(f"{path}_scaler.pkl")
        return model
