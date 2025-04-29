import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

st.title("📈 Sales Forecasting and Model Comparison")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if 'Month' not in data.columns or 'Sales' not in data.columns:
        st.error("The CSV must contain 'Month' and 'Sales' columns.")
    else:
        data.set_index('Month', inplace=True)

        # Split into training and testing sets
        train = data.iloc[:int(0.75 * len(data))]
        test = data.iloc[int(0.75 * len(data)) :]

        # ARIMA
        arima_model = ARIMA(data['Sales'], order=(1,1,1))
        arima_model_fit = arima_model.fit()
        forecast_arima = arima_model_fit.forecast(steps=len(test))

        # Metrics for ARIMA
        arima_mse = mean_squared_error(test['Sales'], forecast_arima)
        arima_rmse = np.sqrt(arima_mse)
        arima_r2 = r2_score(test['Sales'], forecast_arima)

        st.subheader("ARIMA Forecast")
        st.write(f"**MSE:** {arima_mse:.2f}, **RMSE:** {arima_rmse:.2f}, **R²:** {arima_r2:.2f}")
        fig, ax = plt.subplots()
        ax.plot(data['Sales'], label='Historical Sales')
        ax.plot(range(data.index[-1] + 1, data.index[-1] + 1 + len(test)), forecast_arima, 
                label='ARIMA Forecast', linestyle='--', marker='o')
        ax.set_xlabel('Month')
        ax.set_ylabel('Sales')
        ax.legend()
        st.pyplot(fig)

        # Regression models
        X_train = train.index.values.reshape(-1, 1)
        y_train = train['Sales']
        X_test = test.index.values.reshape(-1, 1)
        y_test = test['Sales']

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_predictions = lr_model.predict(X_test)

        lr_mse = mean_squared_error(y_test, lr_predictions)
        lr_rmse = np.sqrt(lr_mse)
        lr_r2 = r2_score(y_test, lr_predictions)

        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)

        rf_mse = mean_squared_error(y_test, rf_predictions)
        rf_rmse = np.sqrt(rf_mse)
        rf_r2 = r2_score(y_test, rf_predictions)

        st.subheader("Model Comparison: Linear Regression vs Random Forest")

        st.write("**Linear Regression**")
        st.write(f"MSE: {lr_mse:.2f}, RMSE: {lr_rmse:.2f}, R²: {lr_r2:.2f}")

        st.write("**Random Forest**")
        st.write(f"MSE: {rf_mse:.2f}, RMSE: {rf_rmse:.2f}, R²: {rf_r2:.2f}")

        fig2, ax2 = plt.subplots()
        ax2.plot(test.index, y_test, label='Actual Sales', marker='x')
        ax2.plot(test.index, lr_predictions, label='Linear Regression', linestyle='--', marker='o')
        ax2.plot(test.index, rf_predictions, label='Random Forest', linestyle='-.', marker='d')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Sales')
        ax2.legend()
        st.pyplot(fig2)

