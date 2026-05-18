# main.py
import os
from data_preprocessing import load_data, add_features, split_and_scale
from linear_regression_model import run_linear_regression
from lstm_model import run_lstm

if not os.path.exists("results"):
    os.makedirs("results")

# Step 1: Load + prepare data
df = load_data("MSFT")
df = add_features(df)
X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

# Step 2: Run Linear Regression
lr_model, lr_mse, lr_mae, lr_dir = run_linear_regression(X_train, X_test, y_train, y_test)

# Step 3: Run LSTM
lstm_model, lstm_mse, lstm_mae, lstm_dir = run_lstm(X_train, X_test, y_train, y_test, window=10)

# Step 4: Compare results
print("\n=== Model Comparison ===")
print(f"Linear RMSE: {lr_mse**0.5:.4f} | MAE: {lr_mae:.4f} | Direction Acc: {lr_dir:.3f}")
print(f"LSTM RMSE: {lstm_mse**0.5:.4f} | MAE: {lstm_mae:.4f} | Direction Acc: {lstm_dir:.3f}")
