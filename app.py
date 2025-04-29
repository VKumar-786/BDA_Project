# Full Google Colab script for ARIMA and model comparison (Linear Regression vs Random Forest)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from google.colab import files

# Upload the dataset file
print("Please upload your dataset CSV file...")
uploaded = files.upload()

# Read the dataset
filename = list(uploaded.keys())[0]
data = pd.read_csv(filename)

# Set 'Month' as index for ARIMA
data.set_index('Month', inplace=True)

# Split into training and testing sets (75% training, 25% testing)
train = data.iloc[:int(0.75 * len(data))]
test = data.iloc[int(0.75 * len(data)):]

# 1. ARIMA Forecasting
arima_model = ARIMA(data['Sales'], order=(1,1,1))  # ARIMA(1,1,1)
arima_model_fit = arima_model.fit()
forecast_arima = arima_model_fit.forecast(steps=6)

# ARIMA Plot
plt.figure(figsize=(10,6))
plt.plot(data['Sales'], label='Historical Sales')
plt.plot(range(25, 31), forecast_arima, label='ARIMA Predictions', linestyle='--', marker='o')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('ARIMA Forecasting for Sales')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation Metrics for ARIMA
arima_mse = mean_squared_error(test['Sales'], forecast_arima)
arima_rmse = np.sqrt(arima_mse)
arima_r2 = r2_score(test['Sales'], forecast_arima)

print(f"\nARIMA Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {arima_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {arima_rmse:.2f}")
print(f"R-squared (R²): {arima_r2:.2f}")

# 2. Comparing Linear Regression vs Random Forest Regression
# Use the index values (Month) as features for regression
X_train = train.index.values.reshape(-1, 1)  # Reshape to a 2D array for sklearn
y_train = train['Sales']
X_test = test.index.values.reshape(-1, 1)
y_test = test['Sales']

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Evaluation Metrics for Linear Regression
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_predictions)

print(f"\nLinear Regression Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {lr_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {lr_rmse:.2f}")
print(f"R-squared (R²): {lr_r2:.2f}")

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest Regression
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"\nRandom Forest Regression Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {rf_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.2f}")
print(f"R-squared (R²): {rf_r2:.2f}")

# Comparison Plot
# Use the index values (Month) for plotting
plt.figure(figsize=(10,6))
plt.plot(test.index, y_test, label='Actual Sales', marker='x')  
plt.plot(test.index, lr_predictions, label='Linear Regression Predictions', linestyle='--', marker='o')
plt.plot(test.index, rf_predictions, label='Random Forest Predictions', linestyle='-.', marker='d') 
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Comparison of Linear Regression and Random Forest Regression')
plt.legend()
plt.grid(True)
plt.show()
