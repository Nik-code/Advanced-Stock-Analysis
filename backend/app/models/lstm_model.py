import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from .model_utils import create_sequences, train_test_split


class LSTMModel:
    def __init__(self, sequence_length=60, units=50, dropout=0.2, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_model(self, input_shape):
        self.model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(units=self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(units=1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')

    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        x, y = create_sequences(scaled_data, self.sequence_length)
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        return x_train, x_test, y_train, y_test

    def train(self, x_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split, callbacks=[early_stopping])

    def predict(self, x):
        return self.scaler.inverse_transform(self.model.predict(x))

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return {'mse': mse, 'rmse': rmse}