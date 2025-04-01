pip uninstall tensorflow

import tensorflow as tf
print(tf.__version__)


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Load the saved model and scaler
model_filename = 'lstm_model.joblib'
scaler_filename = 'scaler.joblib'

model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Function to load and preprocess data
def load_data():
    # Load your dataset
    df = pd.read_csv('CTDAPD Dataset.csv')
    
    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Handle missing values (only for numeric columns)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    return df

# Function to create dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Streamlit application
def app():
    # Load data
    df = load_data()

    # Select 'Anomaly_Score' as the target variable
    target_column = 'Anomaly_Score'
    data = df[target_column]
    
    # Normalize the data using the saved scaler
    data_scaled = scaler.transform(data.values.reshape(-1, 1))
    
    # Prepare data for LSTM model
    time_step = 30
    X, y = create_dataset(data_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data into train and test sets
    train_size = int(len(X) * 0.80)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Predict the future values (for the next 1 year)
    n_days = 365
    predicted_values = []

    last_data = X[-1]
    for i in range(n_days):
        predicted_value = model.predict(last_data.reshape(1, time_step, 1))
        predicted_values.append(predicted_value[0][0])
        last_data = np.append(last_data[1:], predicted_value, axis=0)

    # Invert the scaling to get the actual values
    predicted_values_rescaled = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
    
    # Evaluate model performance (MAE and RMSE)
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = math.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    
    # Display results
    st.title('Anomaly Score Prediction')
    
    # Plot and display actual vs predicted values
    st.subheader('Actual vs Predicted Anomaly Scores')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-len(y_test_rescaled):], y_test_rescaled, label='Actual Anomaly Scores')
    ax.plot(df.index[-len(y_pred_rescaled):], y_pred_rescaled, label='Predicted Anomaly Scores', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Actual vs Predicted Anomaly Scores')
    ax.legend()
    st.pyplot(fig)

    # Plot and display residuals
    residuals = y_test_rescaled.flatten() - y_pred_rescaled.flatten()
    st.subheader('Residuals between Actual and Predicted Anomaly Scores')
    fig_residuals, ax_residuals = plt.subplots(figsize=(12, 6))
    ax_residuals.plot(df.index[-len(residuals):], residuals, label='Residuals', color='purple')
    ax_residuals.set_xlabel('Date')
    ax_residuals.set_ylabel('Residuals')
    ax_residuals.set_title('Residuals between Actual and Predicted Anomaly Scores')
    ax_residuals.legend()
    st.pyplot(fig_residuals)

    # Display MAE and RMSE
    st.subheader('Model Performance')
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"Root Mean Squared Error: {rmse:.4f}")

    # Display predicted anomaly scores for the next 365 days
    st.subheader('Predicted Anomaly Scores for Next 365 Days')
    st.write(predicted_values_rescaled)

if __name__ == "__main__":
    app()
