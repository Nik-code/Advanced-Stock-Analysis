import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)


def train_test_split(x, y, test_size=0.2, random_state=42):
    return sklearn_train_test_split(x, y, test_size=test_size, random_state=random_state)
