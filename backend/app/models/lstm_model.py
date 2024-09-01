import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class LSTMStockPredictor:
    def __init__(self, input_shape, lstm_units=50, dropout_rate=0.2):
        self.model = Sequential()
        self.model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units=lstm_units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def preprocess_data(self, data, sequence_length=60):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaler

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping]
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return mae, rmse

    def plot_predictions(self, y_test, predictions):
        plt.figure(figsize=(14, 5))
        plt.plot(y_test, color='blue', label='Actual Stock Price')
        plt.plot(predictions, color='red', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)


# Example usage:
# data = pd.read_csv('/Users/priyanshsingh/Developer/Projects/Advanced-Stock-Analysis/backend/data/500027.csv')
# predictor = LSTMStockPredictor(input_shape=(60, 1))
# X, y, scaler = predictor.preprocess_data(data['Close Price'].values.reshape(-1, 1))
# history = predictor.train(X, y)
# predictions = predictor.predict(X)
# mae, rmse = predictor.evaluate(X, y)
# predictor.plot_predictions(y, predictions)
