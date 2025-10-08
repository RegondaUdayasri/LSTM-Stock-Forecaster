# 1_train_model.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import save_model
import pickle
import os
# --- Configuration ---
TICKER = 'TSLA' # Change to your preferred stock ticker
START_DATE = '2015-01-01'
END_DATE = '2024-01-01'
LOOK_BACK = 60 # Number of previous days to use as a window

# Create directory to store saved model and scaler
if not os.path.exists('saved_artifacts'):
    os.makedirs('saved_artifacts')


# 1. Data Acquisition and Preprocessing
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
data = df['Close'].values.reshape(-1, 1)

# Initialize and fit the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save the scaler object (Essential for de-scaling predictions later!)
with open('saved_artifacts/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create the training dataset (Sliding Window)
training_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:training_data_len, :]

def create_dataset(dataset, look_back=LOOK_BACK):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data)

# Reshape input to be 3D: [samples, timesteps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 2. Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# 3. Compile and Train
model.compile(optimizer='adam', loss='mean_squared_error')
print("--- Training LSTM Model ---")
# Lower epochs for quicker test, increase to 25-50 for a production model
model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1) 
print("--- Training Complete ---")

# 4. Save the Model
save_model(model, 'saved_artifacts/lstm_model.h5')
print("Model and Scaler saved successfully in 'saved_artifacts/'")
