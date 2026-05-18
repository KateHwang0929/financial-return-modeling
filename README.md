# Financial Return Modeling

This repository contains the code implementation for my research project: **Evaluating the Accuracy of LSTM Neural Networks and Linear Regression for Microsoft Stock Prediction**

The project compares a traditional machine learning model  (Linear Regression)  with a deep learning model (Long Short-Term Memory) to evaluate their performance in predicting Microsoft stock returns using historical market data.

## Research Paper

The full research paper is available on ResearchGate: https://www.researchgate.net/publication/403146749_Evaluating_the_Accuracy_of_LSTM_Neural_Networks_and_Linear_Regression_for_Microsoft_Stock_Prediction


## Project Overview

Stock market prediction is a challenging task because financial data is noisy, nonlinear, and affected by many external factors. This project focuses on Microsoft Corporation (MSFT) and tests whether an LSTM neural network can provide better predictive performance than a baseline Linear Regression model.

The project uses historical MSFT stock data downloaded from Yahoo Finance and applies feature engineering, scaling, model training, prediction, and performance comparison.

## Research Question

Can an LSTM neural network predict stock returns more accurately than a Linear Regression model?

## Models Compared

### Linear Regression

Linear Regression is used as the baseline model. It assumes a linear relationship between engineered stock market features and the next-day return.

### LSTM Neural Network

The LSTM model is used to capture sequential patterns in stock market data. Since stock prices and returns are time-series data, LSTM is tested as a deep learning approach that may better model temporal dependencies.

## Dataset

The dataset consists of historical Microsoft stock data from Yahoo Finance.

The raw stock variables include:

- Open price
- High price
- Low price
- Close price
- Trading volume

The data range used in the project is:

- Training period: 2012 to 2023
- Testing period: 2024

## Feature Engineering

The following features are created and used for prediction:

- Daily return
- 5-day simple moving average
- 10-day simple moving average
- 5-day volatility
- Trading volume

The target variable is the next-day stock return.

## Methodology

The project follows these steps:

1. Download historical MSFT stock data from Yahoo Finance.
2. Clean the dataset and remove missing values.
3. Create engineered financial features.
4. Split the data into training and testing periods.
5. Standardize the input features.
6. Train a Linear Regression model.
7. Train an LSTM neural network.
8. Evaluate both models using error metrics and directional accuracy.
9. Compare the models based on prediction performance.

## Evaluation Metrics

The models are evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Directional Accuracy

Directional accuracy measures whether the model correctly predicts the direction of the next-day return, meaning whether the return is positive or negative.

## Project Structure

```text
financial-return-modeling/
├── main.py
├── data_preprocessing.py
├── linear_regression_model.py
├── lstm_model.py
├── README.md
└── results/
