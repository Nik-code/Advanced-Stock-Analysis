import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


class GRUModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            GRU(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, data, lookback=60):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def train(self, data, epochs=100, batch_size=32):
        X, y = self.prepare_data(data)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, data, steps=7):
        X, _ = self.prepare_data(data)
        last_sequence = X[-1]
        predictions = []
        for _ in range(steps):
            prediction = self.model.predict(last_sequence.reshape(1, -1, 1))
            predictions.append(prediction[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = prediction[0, 0]
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
