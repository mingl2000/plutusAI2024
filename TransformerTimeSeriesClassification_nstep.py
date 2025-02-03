import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------------
# 1. Download Yahoo Stock Data
# -------------------------------
ticker = "000063.sz"  # Change ticker if needed
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
data_pct = np.clip(data_pct, -10, 10)

# -------------------------------
# 3. Create Sequences and Digitize Targets for Multi-Step Prediction
# -------------------------------
def create_sequences(data, seq_length, prediction_steps):
    """
    For each sliding window of length `seq_length`, use the window as input.
    The target is a sequence of the next `prediction_steps` days' price changes
    (for the 'Close' column) digitized into 21 classes.
    
    Digitization:
      - Round the percentage change to nearest integer,
      - Clip to [-10, 10],
      - Shift by +10 so that -10 -> 0, 0 -> 10, +10 -> 20.
    """
    X, y = [], []
    close_index = features.index("Close")  # index of 'Close' in data
    # Ensure there are enough future days for prediction_steps
    for i in range(len(data) - seq_length - prediction_steps + 1):
        seq = data[i : i + seq_length]  # input sequence (seq_length, num_features)
        target_seq = []
        for j in range(prediction_steps):
            pct_change = data[i + seq_length + j, close_index]
            pct_change = np.clip(np.round(pct_change), -10, 10)
            label = int(pct_change + 10)  # shift to range 0..20
            target_seq.append(label)
        X.append(seq)
        y.append(target_seq)
    return np.array(X), np.array(y)

seq_length = 30         # Use past 30 days as input
prediction_steps = 5    # Predict next 5 days (multi-step prediction)
X, y = create_sequences(data_pct, seq_length, prediction_steps)
print("Input shape:", X.shape)   # (num_samples, seq_length, num_features)
print("Target shape:", y.shape)  # (num_samples, prediction_steps), each value in {0,...,20}

# Convert sequences and targets to PyTorch tensors.
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.long)  # shape: (num_samples, prediction_steps)

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
        # Output layer: from d_model to prediction_steps * num_classes
        self.output_linear = nn.Linear(d_model, prediction_steps * num_classes)
        self.num_classes = num_classes
        
    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_length, feature_size)
        Returns:
            Tensor: Logits of shape (batch_size, prediction_steps, num_classes)
        """
        x = self.input_linear(src)  # (batch_size, seq_length, d_model)
        x = x.transpose(0, 1)       # (seq_length, batch_size, d_model)
        encoded = self.transformer_encoder(x)  # (seq_length, batch_size, d_model)
        last_encoded = encoded[-1]  # (batch_size, d_model)
        out = self.output_linear(last_encoded)  # (batch_size, prediction_steps * num_classes)
        out = out.view(-1, self.prediction_steps, self.num_classes)  # reshape
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
epochs = 5
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
        batch_y = y_train[indices]  # (batch_size, prediction_steps)
        
        optimizer.zero_grad()
        logits = model(batch_x)     # (batch_size, prediction_steps, num_classes)
        # Reshape for loss: merge batch and prediction_steps dimensions
        logits_reshaped = logits.view(-1, num_classes)  # (batch_size * prediction_steps, num_classes)
        batch_y_reshaped = batch_y.view(-1)               # (batch_size * prediction_steps)
        loss = criterion(logits_reshaped, batch_y_reshaped)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# -------------------------------
# 7. Inference / Prediction and Plotting (Using First Prediction Step)
# -------------------------------
model.eval()
with torch.no_grad():
    # Compute predictions on the entire training set.
    all_logits = model(X_train)  # shape: (num_samples, prediction_steps, num_classes)
    # For simplicity, we use the first prediction step for plotting.
    predicted_classes = torch.argmax(all_logits[:, 0, :], dim=1)  # (num_samples,)
    
# Convert predicted classes to percentage changes for step 1
predicted_pct = predicted_classes.float() - 10.0  # class 0 -> -10%, 20 -> +10%
# Similarly, ground truth for the first prediction step:
true_pct = y_train[:, 0].float() - 10.0

# Compute prediction error (difference in percentage points)
error = predicted_pct - true_pct

# For sign-based evaluation (using first prediction step)
pred_sign = torch.sign(predicted_pct)
true_sign = torch.sign(true_pct)
correct = (pred_sign == true_sign)
correct_int = correct.int().numpy()
sign_error_rate = 100 * (1 - np.mean(correct_int))

# For plotting, we use a subset (e.g., first 200 samples)
num_plot = 200
indices = np.arange(num_plot)

plt.figure(figsize=(16, 12))

# Plot 1: Predicted vs. Actual Percentage Changes (for first prediction step)
plt.subplot(3, 1, 1)
plt.plot(indices, true_pct[:num_plot].numpy(), label="Actual Price Change (%)", marker="o")
plt.plot(indices, predicted_pct[:num_plot].numpy(), label="Predicted Price Change (%)", marker="x")
plt.title("Predicted vs. Actual Price Movements (Step 1)")
plt.xlabel("Sample Index")
plt.ylabel("Price Change (%)")
plt.legend()

# Plot 2: Prediction Error (Predicted - Actual) for Step 1
plt.subplot(3, 1, 2)
plt.plot(indices, error[:num_plot].numpy(), label="Prediction Error (%)", color="red", marker="d")
plt.axhline(0, color="black", linestyle="--")
plt.title("Prediction Error (Step 1): Predicted - Actual")
plt.xlabel("Sample Index")
plt.ylabel("Error (%)")
plt.legend()

# Plot 3: Sign-Based Correct/Failed Predictions for Step 1
plt.subplot(3, 1, 3)
# Plot a green dot for a correct sign prediction and a red dot for a failed prediction.
for i in range(num_plot):
    if correct_int[i] == 1:
        plt.scatter(i, 0, color="green", s=50)
    else:
        plt.scatter(i, 0, color="red", s=50)
plt.axhline(0, color="black", linestyle="--")
plt.title(f"Sign-Based Prediction (Step 1): Error Rate = {sign_error_rate:.2f}% (Green: Correct, Red: Failed)")
plt.xlabel("Sample Index")
plt.yticks([])  # Hide y-axis ticks

plt.tight_layout()
plt.show()
