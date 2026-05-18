# linear_regression_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from time import time
import matplotlib.pyplot as plt

def run_linear_regression(X_train, X_test, y_train, y_test):
    start = time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_time = time() - start

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    direction_acc = np.mean(np.sign(y_test) == np.sign(y_pred))

    print(f"Linear Regression Results:")
    print(f"RMSE = {mse**0.5:.4f} | MAE = {mae:.4f} | Direction Acc = {direction_acc:.3f}")
    print(f"Training Time: {train_time:.3f}s")

    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, y_pred, label="Predicted")
    plt.title("MSFT Linear Regression: Predicted vs Actual Returns")
    plt.legend(); plt.tight_layout()
    plt.savefig("results/MSFT_linear.png")
    plt.close()

    return model, mse, mae, direction_acc
