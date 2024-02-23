import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load the dataset
def load_dataset(filename):
    try:
        series = pd.read_csv(filename, header=0, index_col=0)
        return series
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None

# Convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols = []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.dropna(inplace=True)
    return agg

# Split into train and test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# Fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    trainX, trainy = train[:, :-1], train[:, -1]
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    yhat = model.predict(np.array([testX]))
    return yhat[0]

# Walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = []
    train, test = train_test_split(data, n_test)
    for i in range(len(test)):
        testX, testy = test[i, :-1], test[i, -1]
        yhat = xgboost_forecast(train, testX)
        predictions.append(yhat)
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    error = mean_absolute_error(test[:, -1], predictions)
    print('Test MAE: %.3f' % error)
    return error, test[:, -1], predictions

# Load the dataset
series = load_dataset('daily-total-female-births.csv')
if series is not None:
    # Prepare data
    values = series.values
    data = series_to_supervised(values, n_in=6).values
    # Evaluate
    mae, y, yhat = walk_forward_validation(data, 12)
    # Plot expected vs predicted values
    plt.figure(figsize=(10,6))
    plt.plot(y, label='Expected')
    plt.plot(yhat, label='Predicted')
    plt.legend()
    plt.show()
