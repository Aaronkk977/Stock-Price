import os

os.environ.update({
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "TF_NUM_INTRAOP_THREADS": "4",
    "TF_NUM_INTEROP_THREADS": "4",
    "OMP_NUM_THREADS": "2",
    "TF_XLA_FLAGS": "--tf_xla_enable_xla_devices=false",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_DATA_THREADPOOL_SIZE_OVERRIDE": "2"
})

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("TensorFlow Version:", tf.__version__)

# ---------------------------------------------------------------------------
# Configuration Parameters (from Paper)
# ---------------------------------------------------------------------------
TICKER = 'BTC-USD'
DATA_PERIOD = "730d" # Fetch slightly more to ensure enough hourly data after potential gaps
DATA_INTERVAL = "1h"
SEQUENCE_LENGTH = 3 # Input: Use previous 3 hours [cite: 110, 132]
FORECAST_HORIZON = 1 # Output: Predict next 1 hour [cite: 110]
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume'] # Adjusted for yfinance auto_adjust=True [cite: 114]
TARGETS = ['High', 'Low'] # Paper predicts High and Low prices
TRAIN_SPLIT_RATIO = 0.8 # [cite: 117]

# N-BEATS Hyperparameters [cite: 143, 154]
N_BLOCKS = 8
N_NEURONS = 256 # Units per block
N_STACKS = 1 # Paper doesn't explicitly mention stacks, assuming 1 stack of N_BLOCKS
N_LAYERS_PER_BLOCK = 4 # Common N-BEATS setting, not specified in paper
THETA_SIZE = N_NEURONS # Size of the expansion coefficients, can be tuned
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

print("\n--- Step 1: Data Acquisition ---")
# Use auto_adjust=True (default), which provides adjusted prices
# and removes 'Adj Close' column.
data_raw = yf.download(TICKER, period=DATA_PERIOD, interval=DATA_INTERVAL)

# --- DIAGNOSTIC PRINT 1 ---
print("\n--- After yf.download() ---")
print(f"Raw data shape: {data_raw.shape}")
print(f"Raw data columns: {data_raw.columns}")
print("Raw Data Head:\n", data_raw.head())
# --- END DIAGNOSTIC PRINT 1 ---

# Check if data is empty
if data_raw.empty:
    raise ValueError(f"Failed to download data for {TICKER}. DataFrame is empty.")

# Select features (Volume might be NaN sometimes, handle later)
# Check if FEATURES exist before selecting
missing_cols = [col for col in FEATURES if col not in data_raw.columns]
if missing_cols:
     raise KeyError(f"Columns {missing_cols} not found in downloaded data. Available columns: {data_raw.columns}")

data = data_raw[FEATURES].copy() # Use .copy() to avoid SettingWithCopyWarning later

# --- DIAGNOSTIC PRINT 2 ---
print("\n--- After Selecting FEATURES ---")
print(f"Selected data shape: {data.shape}")
print(f"Selected data columns: {data.columns}")
print("Selected Data Head:\n", data.head())
# --- END DIAGNOSTIC PRINT 2 ---

# --- Replace dropna with boolean indexing ---
essential_price_cols = ['Open', 'High', 'Low', 'Close']
# Check if essential columns exist before filtering
missing_essential = [col for col in essential_price_cols if col not in data.columns]
if missing_essential:
    raise KeyError(f"Essential columns {missing_essential} are missing before filtering NaNs. Data columns: {data.columns}")

print(f"\nShape before removing NaNs in essential columns: {data.shape}")
# Keep rows where ALL essential price columns are NOT NaN
data = data[data[essential_price_cols].notna().all(axis=1)]
print(f"Shape after removing NaNs in essential columns: {data.shape}")
# --- End Replace dropna ---


# Fill NaN volume with 0, common practice
if 'Volume' in data.columns:
    data['Volume'] = data['Volume'].fillna(0)
else:
    print("Warning: 'Volume' column not found after NaN filtering.")


print(f"\nFinal data shape for preprocessing: {data.shape}")
if data.empty:
    raise ValueError("DataFrame became empty after removing NaNs. Check downloaded data quality.")

# --- End Revised Step 1 ---


print("\n--- Step 2: Data Preprocessing ---")

# Scaling - Fit only on training data
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Split data chronologically BEFORE scaling
split_index = int(TRAIN_SPLIT_RATIO * len(data))
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Fit scalers ONLY on training data
scaled_train_features = feature_scaler.fit_transform(train_data[FEATURES])
# Fit target scaler on High/Low columns of training data
target_scaler.fit(train_data[TARGETS])

# Transform train and test data using fitted scalers
scaled_test_features = feature_scaler.transform(test_data[FEATURES])

# Combine scaled features into DataFrames for easier indexing
scaled_train_df = pd.DataFrame(scaled_train_features, columns=FEATURES, index=train_data.index)
scaled_test_df = pd.DataFrame(scaled_test_features, columns=FEATURES, index=test_data.index)

# Also scale the target columns separately for sequence creation
scaled_train_targets_df = pd.DataFrame(target_scaler.transform(train_data[TARGETS]), columns=TARGETS, index=train_data.index)
scaled_test_targets_df = pd.DataFrame(target_scaler.transform(test_data[TARGETS]), columns=TARGETS, index=test_data.index)


# Sequence Creation Function [cite: 132]
def create_sequences(input_features_df, input_targets_df, sequence_length, forecast_horizon):
    X, y = [], []
    indices = []
    # Ensure target indices align with the end of the input sequence
    for i in range(len(input_features_df) - sequence_length - forecast_horizon + 1):
        feature_sequence = input_features_df.iloc[i:(i + sequence_length)].values
        target_values = input_targets_df.iloc[i + sequence_length : i + sequence_length + forecast_horizon].values

        # Ensure the target shape is (forecast_horizon, num_targets) -> (1, 2)
        if target_values.shape == (forecast_horizon, len(TARGETS)):
           X.append(feature_sequence)
           y.append(target_values.flatten()) # Flatten to (2,) for Dense layer output
           indices.append(input_targets_df.index[i + sequence_length]) # Store index of the target time step
        # else: # Debugging shape issues
            # print(f"Skipping index {i} due to shape mismatch: Target shape {target_values.shape}")

    return np.array(X), np.array(y), indices

# Create sequences for training and testing sets
X_train, y_train, train_indices = create_sequences(scaled_train_df, scaled_train_targets_df, SEQUENCE_LENGTH, FORECAST_HORIZON)
X_test, y_test, test_indices = create_sequences(scaled_test_df, scaled_test_targets_df, SEQUENCE_LENGTH, FORECAST_HORIZON)

print(f"X_train shape: {X_train.shape}") # Should be (num_samples, 3, num_features)
print(f"y_train shape: {y_train.shape}") # Should be (num_samples, 2)
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Verify shapes
num_features = len(FEATURES)
num_targets = len(TARGETS)
if X_train.shape[1:] != (SEQUENCE_LENGTH, num_features):
    raise ValueError(f"X_train shape mismatch: expected (None, {SEQUENCE_LENGTH}, {num_features}), got {X_train.shape}")
if y_train.shape[1:] != (num_targets,):
     raise ValueError(f"y_train shape mismatch: expected (None, {num_targets}), got {y_train.shape}")

print("\n--- Step 3: N-BEATS Model Definition ---")

# N-BEATS Block Layer Implementation (Same as before)
class NBeatsBlock(layers.Layer):
    def __init__(self, input_size, theta_size, horizon, n_neurons, n_layers, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.theta_size = theta_size
        # Fully connected layers (stack)
        self.hidden = [layers.Dense(n_neurons, activation='relu') for _ in range(n_layers)]
        # Output layers for basis coefficients (thetas)
        self.theta_f = layers.Dense(theta_size, activation='linear', name='theta_f') # Forecast
        self.theta_b = layers.Dense(theta_size, activation='linear', name='theta_b') # Backcast

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta_f = self.theta_f(x)
        theta_b = self.theta_b(x)
        # Simplified basis expansion - direct use of thetas
        # Final projection handled in the main model build
        return theta_b, theta_f

# Build the full N-BEATS model with residual connections (Revised forecast_sum handling)
def create_nbeats_model(seq_len, horizon, num_features, num_targets, n_blocks, n_neurons, n_layers_per_block, theta_size):
    input_layer = layers.Input(shape=(seq_len, num_features), name='model_input')
    # Flatten the input sequence for Dense layers
    x_flat = layers.Flatten()(input_layer)

    residuals = x_flat
    forecast_sum = None # Initialize as None

    for i in range(n_blocks):
        block = NBeatsBlock(input_size=seq_len * num_features, # Block operates on flattened input
                            theta_size=theta_size,
                            horizon=horizon,
                            n_neurons=n_neurons,
                            n_layers=n_layers_per_block,
                            name=f'nbeats_block_{i}')
        # Get backcast and forecast thetas from the block
        theta_b, theta_f = block(residuals)

        # Simplified backcast/forecast generation
        backcast_layer = layers.Dense(seq_len * num_features, name=f'backcast_{i}')
        backcast = backcast_layer(theta_b)

        forecast_layer = layers.Dense(horizon * num_targets, name=f'forecast_{i}', activation='sigmoid')
        forecast = forecast_layer(theta_f) # Shape (batch_size, horizon * num_targets)

        # Double Residual Connection: Subtract backcast from input to block
        residuals = layers.subtract([residuals, backcast], name=f'subtract_{i}')

        # Add the block's forecast to the total forecast
        if forecast_sum is None:
            # Initialize with the first block's forecast
            forecast_sum = forecast
        else:
            # Add subsequent forecasts
            forecast_sum = layers.add([forecast_sum, forecast], name=f'add_{i}')

    # Ensure final output shape is (batch_size, num_targets) for horizon=1
    if horizon == 1:
      # If horizon is 1, forecast_sum already has shape (batch_size, num_targets)
      final_forecast = forecast_sum
    else:
      # If horizon > 1, reshape might be needed
      final_forecast = layers.Reshape((horizon, num_targets), name='final_forecast')(forecast_sum)
      # If you need only the first step of a multi-step forecast:
      # final_forecast = layers.Lambda(lambda x: x[:, 0, :], name='first_step_forecast')(final_forecast)


    model = tf.keras.Model(inputs=input_layer, outputs=final_forecast, name='NBEATS_model')
    return model

# Instantiate the model
model = create_nbeats_model(
    seq_len=SEQUENCE_LENGTH,
    horizon=FORECAST_HORIZON,
    num_features=num_features,
    num_targets=num_targets,
    n_blocks=N_BLOCKS,
    n_neurons=N_NEURONS,
    n_layers_per_block=N_LAYERS_PER_BLOCK,
    theta_size=THETA_SIZE
)

# Compile the model
model.compile(loss='mae', # Mean Absolute Error
              optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=['mae'])

model.summary()

print("\n--- Step 4: Model Training ---")

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=10, # Stop if no improvement for 10 epochs
                                                  restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.2, # Reduce LR by factor of 5
                                                 patience=5,
                                                 min_lr=LEARNING_RATE / 100)

# Train the model
def build_dataset(X, y, batch):
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch)
    opts = tf.data.Options()
    opts.threading.private_threadpool_size = 2
    opts.threading.max_intra_op_parallelism = 1
    return ds.with_options(opts).prefetch(1)

history = model.fit(
    build_dataset(X_train, y_train, batch=256),
    epochs=EPOCHS,
    validation_data=build_dataset(X_test, y_test, batch=256),
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss History')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.legend()
plt.savefig('model_loss_history.png')

print("\n--- Step 5: Evaluation ---")

# Make predictions on the test set
predictions_scaled = model.predict(X_test)
predictions_scaled = predictions_scaled.clip(0, 1)  # Clip predictions to [0, 1] range

# Inverse transform predictions and actual values
# Predictions shape: (num_samples, num_targets)
# y_test shape: (num_samples, num_targets)
predictions_original = target_scaler.inverse_transform(predictions_scaled)
y_test_original = target_scaler.inverse_transform(y_test)

# Calculate metrics on the original scale
mae = mean_absolute_error(y_test_original, predictions_original)
r2 = r2_score(y_test_original, predictions_original)

print(f"\nTest Set Evaluation (Original Scale):")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R-squared (R2 Score):      {r2:.6f}")

# gpt debug
mae_scaled = mean_absolute_error(y_test, predictions_scaled.clip(0,1))
r2_scaled  = r2_score(y_test, predictions_scaled.clip(0,1))
print(f"Scaled MAE: {mae_scaled:.6f},  Scaled R2: {r2_scaled:.4f}")

# Comparison with paper's results (Note potential ambiguity/typo in paper's reported values)
# Abstract: MAE 0.9998, R2 0.00240 [cite: 7]
# Results : MAE 0.00240, R2 0.9998 [cite: 156, 158]
# Assuming results section values are more likely intended for scaled data
print("\nComparison with Paper (Results Section - likely scaled values):")
print(f"Paper MAE (Scaled?): 0.00240")
print(f"Paper R2  (Scaled?): 0.9998")
print("\nComparison with Paper (Abstract - likely scaled values):")
print(f"Paper MAE (Scaled?): 0.9998")
print(f"Paper R2  (Scaled?): 0.00240")
print("\nNote: The MAE/R2 calculated here are on the *original price scale*,")
print("while the paper's values might be on the *scaled* data (0-1 range).")
print("Direct comparison might be misleading without knowing the paper's exact evaluation scale.")

print("\n--- Step 6: Visualization ---")

# Use the test_indices to align predictions with the original dates if needed
# prediction_dates = test_indices[:len(predictions_original)] # Get corresponding dates

# Plot actual vs predicted High/Low prices for a subset of the test data
plot_points = 500
high_col_index = TARGETS.index('High')
low_col_index = TARGETS.index('Low')

plt.figure(figsize=(15, 7))
plt.plot(y_test_original[:plot_points, high_col_index], label='Actual High', color='blue', linewidth=1)
plt.plot(predictions_original[:plot_points, high_col_index], label='Predicted High', color='orange', linestyle='--', linewidth=1)
plt.title(f'High Price Predictions vs Actual (First {plot_points} Test Points)')
plt.xlabel('Time Steps in Test Set')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('high_price_predictions.png')

plt.figure(figsize=(15, 7))
plt.plot(y_test_original[:plot_points, low_col_index], label='Actual Low', color='blue', linewidth=1)
plt.plot(predictions_original[:plot_points, low_col_index], label='Predicted Low', color='orange', linestyle='--', linewidth=1)
plt.title(f'Low Price Predictions vs Actual (First {plot_points} Test Points)')
plt.xlabel('Time Steps in Test Set')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('low_price_predictions.png')

print("\n--- Script Finished ---")