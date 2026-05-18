import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from time import time
import matplotlib.pyplot as plt

def create_sequences(X, y, window=10):
    Xs, ys = [], []

    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y.iloc[i+window])

    return np.array(Xs), np.array(ys)

def run_lstm(X_train, X_test, y_train, y_test, window=10):
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, window)

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    start = time()

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[es]
    )

    train_time = time() - start

    y_pred = model.predict(X_test_seq).flatten()

    mse = mean_squared_error(y_test_seq, y_pred)
    mae = mean_absolute_error(y_test_seq, y_pred)
    direction_acc = np.mean(np.sign(y_test_seq) == np.sign(y_pred))

    print("LSTM Results")
    print(f"RMSE = {mse**0.5:.4f} | MAE = {mae:.4f} | Direction Acc = {direction_acc:.3f}")
    print(f"Training Time: {train_time:.2f}s")

    plt.plot(y_test_seq, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("MSFT LSTM Predicted vs Actual Returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/MSFT_lstm.png")
    plt.close()

    return model, mse, mae, direction_acc