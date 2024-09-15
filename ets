import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Sample data: replace this with your univariate time series data
data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140]
df = pd.DataFrame(data, columns=['value'])

# Split the data into train and test sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Fit the ETS model (with additive trend and seasonality by default)
model = ExponentialSmoothing(train['value'], seasonal='add', trend='add', seasonal_periods=12).fit()

# Forecast the next periods (same length as the test set)
forecast = model.forecast(steps=len(test))

# Calculate the Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(test, forecast)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(train.index, train['value'], label='Train')
plt.plot(test.index, test['value'], label='Test', color='orange')
plt.plot(test.index, forecast, label='Forecast', color='green')
plt.legend(loc='best')
plt.title("ETS Forecast vs Actual")
plt.show()
