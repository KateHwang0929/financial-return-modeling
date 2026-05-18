# data_preprocessing.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def load_data(ticker="MSFT"):
    """Download daily stock data from Yahoo Finance."""
    df = yf.download(ticker, start="2012-01-01", end="2024-12-31", auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

def add_features(df):
    """Add engineered features: returns, moving averages, volatility."""
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["Volatility_5"] = df["Return"].rolling(5).std()
    df.dropna(inplace=True)
    return df

def split_and_scale(df):
    """Split into train (2012–2023) and test (2024) and scale features."""
    features = ["Return", "SMA_5", "SMA_10", "Volatility_5", "Volume"]
    X = df[features]
    y = df["Return"].shift(-1).dropna()
    X = X.iloc[:-1]  # align features and target

    X_train = X.loc["2012":"2023"]
    X_test  = X.loc["2024":]
    y_train = y.loc["2012":"2023"]
    y_test  = y.loc["2024":]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_data()
    df = add_features(df)
    print(df.head())
