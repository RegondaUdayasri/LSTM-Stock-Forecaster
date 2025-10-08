# 2_app_deploy.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_PATH = 'saved_artifacts/lstm_model.h5'
SCALER_PATH = 'saved_artifacts/scaler.pkl'
LOOK_BACK = 60 # Must match the value used during training!

# --- Load Artifacts ---
@st.cache_resource # Cache the model and scaler to prevent reloading on every interaction
def load_artifacts():
    # Check if files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("Model or Scaler files not found. Please run '1_train_model.py' first.")
        return None, None

    # Load the trained model
    model = load_model(MODEL_PATH)
    
    # Load the scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

model, scaler = load_artifacts()

# --- Streamlit App UI ---
st.title("📈 LSTM Stock Price Prediction App")
st.markdown("A Deep Learning project demonstrating Time Series Forecasting.")

if model is not None:
    # User Input
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL, TSLA)", 'AAPL')
    
    if st.button("Generate Prediction"):
        with st.spinner(f"Fetching data and generating prediction for {ticker}..."):
            try:
                # 1. Fetch Latest Data
                df = yf.download(ticker, period="max")
                
                if df.empty:
                    st.error(f"Could not find data for ticker: {ticker}. Please check the symbol.")
                    st.stop()
                    
                # Use the 'Close' prices
                data = df['Close'].values.reshape(-1, 1)
                
                # 2. Prepare Test Data
                # Get the last LOOK_BACK days for the current prediction
                last_look_back_days = data[-LOOK_BACK:]
                
                # Apply the saved scaler
                last_look_back_days_scaled = scaler.transform(last_look_back_days)
                
                # Reshape for LSTM: [1, timesteps, features]
                X_predict = np.reshape(last_look_back_days_scaled, (1, LOOK_BACK, 1))
                
                # 3. Make Prediction
                scaled_prediction = model.predict(X_predict)
                
                # 4. Inverse Transform to Actual Price
                final_prediction = scaler.inverse_transform(scaled_prediction)
                
                # 5. Display Results
                st.success(f"Successfully generated prediction for **{ticker}**")
                
                # Display the predicted next closing price
                st.metric(label=f"Predicted Next Closing Price for {ticker}", 
                          value=f"${final_prediction[0][0]:,.2f}")

                # Optional: Plot the historical data with the prediction point
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot historical data
                historical_dates = df.index[-200:] # Show last 200 days
                historical_prices = df['Close'][-200:]
                ax.plot(historical_dates, historical_prices, label='Historical Price', color='blue')
                
                # Plot the predicted next day's price
                last_date = df.index[-1]
                predicted_date = last_date + pd.Timedelta(days=1)
                
                # Adjust for weekends/holidays by finding the next market day
                if predicted_date.weekday() >= 5: # Saturday (5) or Sunday (6)
                    days_to_add = 7 - predicted_date.weekday()
                    predicted_date += pd.Timedelta(days=days_to_add)

                ax.scatter(predicted_date, final_prediction[0][0], color='red', s=100, zorder=5)
                ax.text(predicted_date, final_prediction[0][0] * 1.005, 
                        f'Prediction: ${final_prediction[0][0]:.2f}', 
                        fontsize=12, color='red')
                
                ax.set_title(f'{ticker} Stock Price History with Next Day Prediction')
                ax.set_xlabel('Date')
                ax.set_ylabel('Close Price (USD)')
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Model is not loaded. Please run the training script first and ensure files are in 'saved_artifacts/'.")
