import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------------
# 1. Download Yahoo Stock Data
# -------------------------------
ticker = "AAPL"  # Change ticker if needed
df = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# We'll use four features: Open, High, Low, Close.
features = ["Open", "High", "Low", "Close"]

# -------------------------------
# 2. Convert Prices to Daily Percentage Changes (Price Movements)
# -------------------------------
# Compute daily percentage change (in percent) for each feature.
# (pct_change returns NaN for the first row; drop it)
data_pct = df[features].pct_change() * 100  
data_pct = data_pct.dropna().values  # shape: (num_days-1, num_features)

# Clip the percentage changes to the range [-10, 10]
data_pct = np.clip(np.round(data_pct), -10, 10)


# -------------------------------
# 3. Create Sequences and Digitize Targets for Multi-Step Prediction
# -------------------------------
def create_sequences(data, seq_length, prediction_steps):
    """
    For each sliding window of length `seq_length`, use the window as input.
    The target is a sequence of the next `prediction_steps` days' price changes
    (for 'High', 'Low', 'Close' columns) digitized into 21 classes each.
    
    Digitization:
      - Round the percentage change to nearest integer,
      - Clip to [-10, 10],
      - Shift by +10 so that -10 -> 0, 0 -> 10, +10 -> 20.
    """
    X, y = [], []
    high_index = features.index("High")  # index of 'High' in data
    low_index = features.index("Low")   # index of 'Low' in data
    close_index = features.index("Close")  # index of 'Close' in data

    # Ensure there are enough future days for prediction_steps
    for i in range(len(data) - seq_length - prediction_steps + 1):
        seq = data[i : i + seq_length]  # input sequence (seq_length, num_features)
        
        target_seq = []
        for j in range(prediction_steps):
            high_pct = data[i + seq_length + j, high_index]
            low_pct = data[i + seq_length + j, low_index]
            close_pct = data[i + seq_length + j, close_index]
            
            # Digitize each target value
            high_pct = np.clip(np.round(high_pct), -10, 10) + 10  # shift to range 0..20
            low_pct = np.clip(np.round(low_pct), -10, 10) + 10    # shift to range 0..20
            close_pct = np.clip(np.round(close_pct), -10, 10) + 10  # shift to range 0..20
            
            target_seq.append([high_pct, low_pct, close_pct])
        X.append(seq)
        y.append(target_seq)
    
    return np.array(X), np.array(y)

seq_length = 30         # Use past 30 days as input
prediction_steps = 2    # Predict next 2 days (multi-step prediction)
X, y = create_sequences(data_pct, seq_length, prediction_steps)
print("Input shape:", X.shape)   # (num_samples, seq_length, num_features)
print("Target shape:", y.shape)  # (num_samples, prediction_steps, 3), each value in {0,...,20}

# Convert sequences and targets to PyTorch tensors.
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.long)  # shape: (num_samples, prediction_steps, 3)

# -------------------------------
# 4. Define the Multi-Step Transformer-Based Classifier Model
# -------------------------------
class TransformerTimeSeriesMultiStepClassifier(nn.Module):
    def __init__(self, feature_size, d_model=64, nhead=4, num_layers=2, 
                 num_classes=21, prediction_steps=5, dropout=0.1):
        """
        A Transformer encoder model that predicts multiple steps ahead.
        
        Args:
            feature_size (int): Number of features per time step.
            d_model (int): Dimension of model (hidden size).
            nhead (int): Number of heads in multi-head attention.
            num_layers (int): Number of encoder layers.
            num_classes (int): Number of output classes (here, 21 for -10% to +10%).
            prediction_steps (int): Number of future steps to predict.
            dropout (float): Dropout rate.
        """
        super(TransformerTimeSeriesMultiStepClassifier, self).__init__()
        self.prediction_steps = prediction_steps
        self.input_linear = nn.Linear(feature_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output layer: from d_model to prediction_steps * num_classes * 3 (High, Low, Close)
        self.output_linear = nn.Linear(d_model, prediction_steps * num_classes * 3)
        self.num_classes = num_classes
        
    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_length, feature_size)
        Returns:
            Tensor: Logits of shape (batch_size, prediction_steps, 3, num_classes)
        """
        x = self.input_linear(src)  # (batch_size, seq_length, d_model)
        x = x.transpose(0, 1)       # (seq_length, batch_size, d_model)
        encoded = self.transformer_encoder(x)  # (seq_length, batch_size, d_model)
        last_encoded = encoded[-1]  # (batch_size, d_model)
        out = self.output_linear(last_encoded)  # (batch_size, prediction_steps * 3 * num_classes)
        out = out.view(-1, self.prediction_steps, 3, self.num_classes)  # reshape to (batch_size, prediction_steps, 3, num_classes)
        return out

feature_size = len(features)  # 4 features
num_classes = 21
model = TransformerTimeSeriesMultiStepClassifier(feature_size=feature_size, d_model=64,
                                                   nhead=4, num_layers=2,
                                                   num_classes=num_classes,
                                                   prediction_steps=prediction_steps)

# -------------------------------
# 5. Set Up Loss and Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 6. Training Loop
# -------------------------------
epochs = 300
batch_size = 32
num_batches = X_train.shape[0] // batch_size

print("Training started...")
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.shape[0])
    epoch_loss = 0.0

    for i in range(num_batches):
        indices = permutation[i * batch_size:(i + 1) * batch_size]
        batch_x = X_train[indices]  # (batch_size, seq_length, feature_size)
        batch_y = y_train[indices]  # (batch_size, prediction_steps, 3)
        
        optimizer.zero_grad()
        logits = model(batch_x)     # (batch_size, prediction_steps, 3, num_classes)
        
        # Reshape for loss: merge batch and prediction_steps dimensions
        logits_reshaped = logits.view(-1, num_classes)  # (batch_size * prediction_steps * 3, num_classes)
        batch_y_reshaped = batch_y.view(-1, 3)  # (batch_size * prediction_steps, 3)
        
        # Flatten batch_y_reshaped to compute loss for all three targets (High, Low, Close)
        batch_y_reshaped = batch_y_reshaped.view(-1)  # (batch_size * prediction_steps * 3,)
        
        loss = criterion(logits_reshaped, batch_y_reshaped)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# -------------------------------
# 7. Inference / Prediction and Plotting (Using First Prediction Step)
# -------------------------------
# -------------------------------
# 7. Inference / Prediction and Plotting (Using First Prediction Step)
# -------------------------------
# -------------------------------
# 7. Inference / Prediction and Plotting (Using First Prediction Step)
# -------------------------------
model.eval()
with torch.no_grad():
    # Compute predictions on the entire training set.
    all_logits = model(X_train)  # shape: (num_samples, prediction_steps, 3, num_classes)
    
    # For simplicity, we use the first prediction step for plotting.
    predicted_classes = torch.argmax(all_logits[:, 0, :, :], dim=2)  # (num_samples, 3)
    
# Convert predicted classes to percentage changes for step 1
predicted_pct = predicted_classes.float() - 10.0  # class 0 -> -10%, 20 -> +10%
# Similarly, ground truth for the first prediction step:

true_pct = y_train[:, 0, :]-10  # shape: (num_samples, 3)

# Compute prediction error (difference in percentage points)
error_high = predicted_pct[:, 0] - true_pct[:, 0]
error_low = predicted_pct[:, 1] - true_pct[:, 1]
error_close = predicted_pct[:, 2] - true_pct[:, 2]

# For sign-based evaluation
pred_sign_high = torch.sign(predicted_pct[:, 0])
true_sign_high = torch.sign(true_pct[:, 0])

pred_sign_low = torch.sign(predicted_pct[:, 1])
true_sign_low = torch.sign(true_pct[:, 1])

pred_sign_close = torch.sign(predicted_pct[:, 2])
true_sign_close = torch.sign(true_pct[:, 2])

# Correct predictions (sign-based)
correct_high = (pred_sign_high == true_sign_high)
correct_low = (pred_sign_low == true_sign_low)
correct_close = (pred_sign_close == true_sign_close)

correct_int_high = correct_high.int().numpy()
correct_int_low = correct_low.int().numpy()
correct_int_close = correct_close.int().numpy()

# Sign-based error rates
sign_error_rate_high = 100 * (1 - np.mean(correct_int_high))
sign_error_rate_low = 100 * (1 - np.mean(correct_int_low))
sign_error_rate_close = 100 * (1 - np.mean(correct_int_close))

# For plotting, we use a subset (e.g., first 200 samples)
num_plot = 200
indices = np.arange(num_plot)

# Plotting - separate sets for High, Low, and Close
plt.figure(figsize=(16, 18))

# --------------------------
# High Price Predictions
# --------------------------
plt.subplot(3, 1, 1)
plt.plot(indices, true_pct[:num_plot, 0].numpy(), label="Actual High Change (%)", color="blue", marker="o")
plt.plot(indices, predicted_pct[:num_plot, 0].numpy(), label="Predicted High Change (%)", color="red", marker="x")
plt.title("High Price: Predicted vs. Actual Price Movements")
plt.xlabel("Sample Index")
plt.ylabel("Price Change (%)")
plt.legend()

# High Prediction Error
plt.subplot(3, 1, 2)
plt.plot(indices, error_high[:num_plot].numpy(), label="Prediction Error High (%)", color="blue", marker="d")
plt.axhline(0, color="black", linestyle="--")
plt.title("High Price Prediction Error (Predicted - Actual)")
plt.xlabel("Sample Index")
plt.ylabel("Error (%)")
plt.legend()

# High Sign-Based Correct/Failed Predictions
plt.subplot(3, 1, 3)
# Plot a green dot for a correct sign prediction and a red dot for a failed prediction.
for i in range(num_plot):
    if correct_int_high[i] == 1:
        plt.scatter(i, 0, color="blue", s=50)  # High predictions in blue
    else:
        plt.scatter(i, 0, color="red", s=50)
plt.axhline(0, color="black", linestyle="--")
plt.title(f"High Price: Sign-Based Prediction (Error Rate = {sign_error_rate_high:.2f}%)")
plt.xlabel("Sample Index")
plt.yticks([])  # Hide y-axis ticks

plt.tight_layout()
#plt.show()

# --------------------------
# Low Price Predictions
# --------------------------
plt.figure(figsize=(16, 18))

# Low Price Predictions
plt.subplot(3, 1, 1)
plt.plot(indices, true_pct[:num_plot, 1].numpy(), label="Actual Low Change (%)", color="green", marker="o")
plt.plot(indices, predicted_pct[:num_plot, 1].numpy(), label="Predicted Low Change (%)", color="orange", marker="x")
plt.title("Low Price: Predicted vs. Actual Price Movements")
plt.xlabel("Sample Index")
plt.ylabel("Price Change (%)")
plt.legend()

# Low Prediction Error
plt.subplot(3, 1, 2)
plt.plot(indices, error_low[:num_plot].numpy(), label="Prediction Error Low (%)", color="green", marker="d")
plt.axhline(0, color="black", linestyle="--")
plt.title("Low Price Prediction Error (Predicted - Actual)")
plt.xlabel("Sample Index")
plt.ylabel("Error (%)")
plt.legend()

# Low Sign-Based Correct/Failed Predictions
plt.subplot(3, 1, 3)
# Plot a green dot for a correct sign prediction and a red dot for a failed prediction.
for i in range(num_plot):
    if correct_int_low[i] == 1:
        plt.scatter(i, 0, color="green", s=50)  # Low predictions in green
    else:
        plt.scatter(i, 0, color="red", s=50)
plt.axhline(0, color="black", linestyle="--")
plt.title(f"Low Price: Sign-Based Prediction (Error Rate = {sign_error_rate_low:.2f}%)")
plt.xlabel("Sample Index")
plt.yticks([])  # Hide y-axis ticks

plt.tight_layout()
#plt.show()

# --------------------------
# Close Price Predictions
# --------------------------
plt.figure(figsize=(16, 18))

# Close Price Predictions
plt.subplot(3, 1, 1)
plt.plot(indices, true_pct[:num_plot, 2].numpy(), label="Actual Close Change (%)", color="purple", marker="o")
plt.plot(indices, predicted_pct[:num_plot, 2].numpy(), label="Predicted Close Change (%)", color="brown", marker="x")
plt.title("Close Price: Predicted vs. Actual Price Movements")
plt.xlabel("Sample Index")
plt.ylabel("Price Change (%)")
plt.legend()

# Close Prediction Error
plt.subplot(3, 1, 2)
plt.plot(indices, error_close[:num_plot].numpy(), label="Prediction Error Close (%)", color="purple", marker="d")
plt.axhline(0, color="black", linestyle="--")
plt.title("Close Price Prediction Error (Predicted - Actual)")
plt.xlabel("Sample Index")
plt.ylabel("Error (%)")
plt.legend()

# Close Sign-Based Correct/Failed Predictions
plt.subplot(3, 1, 3)
# Plot a green dot for a correct sign prediction and a red dot for a failed prediction.
for i in range(num_plot):
    if correct_int_close[i] == 1:
        plt.scatter(i, 0, color="purple", s=50)  # Close predictions in purple
    else:
        plt.scatter(i, 0, color="red", s=50)
plt.axhline(0, color="black", linestyle="--")
plt.title(f"Close Price: Sign-Based Prediction (Error Rate = {sign_error_rate_close:.2f}%)")
plt.xlabel("Sample Index")
plt.yticks([])  # Hide y-axis ticks

plt.tight_layout()
plt.show()
