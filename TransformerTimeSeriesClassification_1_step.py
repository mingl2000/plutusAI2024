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
df = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# We'll use four features: Open, High, Low, Close.
features = ["Open", "High", "Low", "Close"]

# -------------------------------
# 2. Convert Prices to Daily Percentage Changes (Price Movements)
# -------------------------------
# Compute daily percentage change (in percent) for each feature.
# (pct_change returns NaN for the first row; we drop it)
data_pct = df[features].pct_change() * 100  
data_pct = data_pct.dropna().values  # shape: (num_days-1, num_features)

# Clip the percentage changes to the range [-10, 10]
data_pct = np.clip(data_pct, -10, 10)

# -------------------------------
# 3. Create Sequences and Digitize the Target Price Movement
# -------------------------------
def create_sequences(data, seq_length):
    """
    For each sliding window of length `seq_length`, use the window as input.
    The target is the digitized next-day price change for the 'Close' column.
    Digitization: Round the percentage change, clip to [-10, 10],
    then shift so that -10 -> 0, 0 -> 10, and +10 -> 20.
    """
    X, y = [], []
    close_index = features.index("Close")  # index of Close price in the data
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]  # shape (seq_length, num_features)
        # Next day's "Close" percentage change (target)
        pct_change = data[i + seq_length, close_index]
        # Round to nearest integer (1% steps) and clip to [-10, 10]
        pct_change = np.clip(np.round(pct_change), -10, 10)
        # Shift: so that -10 -> 0, 0 -> 10, 10 -> 20
        label = int(pct_change + 10)
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

seq_length = 30  # Use past 30 days to predict the next day's movement
X, y = create_sequences(data_pct, seq_length)
print("Input shape:", X.shape)   # (num_samples, seq_length, num_features)
print("Target shape:", y.shape)  # (num_samples,), classes in {0,...,20}

# Convert sequences and targets to PyTorch tensors.
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.long)

# -------------------------------
# 4. Define the Transformer-Based Classifier Model
# -------------------------------
class TransformerTimeSeriesClassifier(nn.Module):
    def __init__(self, feature_size, d_model=64, nhead=4, num_layers=2, num_classes=21, dropout=0.1):
        """
        A simple Transformer encoder model for time series classification.
        
        Args:
            feature_size (int): Number of features per time step.
            d_model (int): Dimension of the model (hidden size).
            nhead (int): Number of heads in the multi-head attention.
            num_layers (int): Number of encoder layers.
            num_classes (int): Number of output classes (here, 21 for -10% to +10%).
            dropout (float): Dropout rate.
        """
        super(TransformerTimeSeriesClassifier, self).__init__()
        # Project input features to model dimension
        self.input_linear = nn.Linear(feature_size, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output linear layer maps from d_model to num_classes (21)
        self.output_linear = nn.Linear(d_model, num_classes)
    
    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_length, feature_size)
        Returns:
            Tensor: Logits of shape (batch_size, num_classes)
        """
        # Project input: (batch_size, seq_length, d_model)
        x = self.input_linear(src)
        # Transformer expects input shape: (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)
        # Pass through Transformer encoder
        encoded = self.transformer_encoder(x)
        # Use the last time step's output: (batch_size, d_model)
        last_encoded = encoded[-1]
        # Get class logits: (batch_size, num_classes)
        logits = self.output_linear(last_encoded)
        return logits

feature_size = len(features)  # 4 features
num_classes = 21
model = TransformerTimeSeriesClassifier(feature_size=feature_size, d_model=64, nhead=4, num_layers=2, num_classes=num_classes)

# -------------------------------
# 5. Set Up Loss and Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 6. Training Loop
# -------------------------------
epochs = 200
batch_size = 32
num_batches = X_train.shape[0] // batch_size

print("Training started...")
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.shape[0])
    epoch_loss = 0.0

    for i in range(num_batches):
        indices = permutation[i * batch_size:(i + 1) * batch_size]
        batch_x = X_train[indices]  # shape: (batch_size, seq_length, feature_size)
        batch_y = y_train[indices]  # shape: (batch_size,)
        
        optimizer.zero_grad()
        logits = model(batch_x)     # shape: (batch_size, num_classes)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# -------------------------------
# 7. Inference / Prediction and Plotting
# -------------------------------
model.eval()
with torch.no_grad():
    # Compute predictions on the entire training set.
    all_logits = model(X_train)  # (num_samples, num_classes)
    predicted_classes = torch.argmax(all_logits, dim=1)  # (num_samples,)

# Convert predicted classes to percentage changes
predicted_pct = predicted_classes.float() - 10.0  # Now, class 0 -> -10%, class 20 -> +10%
# Convert ground truth labels to percentage changes
true_pct = y_train.float() - 10.0

# Compute prediction error (difference between predicted and actual percentage change)
error = predicted_pct - true_pct  # Error in percentage points

# For plotting, we use a subset (e.g., first 200 samples)
num_plot = 200
indices = np.arange(num_plot)

plt.figure(figsize=(14, 10))

# Plot 1: Predicted vs. Actual Percentage Changes
plt.subplot(3, 1, 1)
plt.plot(indices, true_pct[:num_plot].numpy(), label="Actual Price Change (%)", marker="o")
plt.plot(indices, predicted_pct[:num_plot].numpy(), label="Predicted Price Change (%)", marker="x")
plt.title("Predicted vs. Actual Price Movements")
plt.xlabel("Sample Index")
plt.ylabel("Price Change (%)")
plt.legend()

# Plot 2: Prediction Error (Predicted - Actual)
plt.subplot(3, 1, 2)
plt.plot(indices, error[:num_plot].numpy(), label="Prediction Error (%)", color="red", marker="d")
plt.axhline(0, color="black", linestyle="--")
plt.title("Prediction Error (Predicted - Actual)")
plt.xlabel("Sample Index")
plt.ylabel("Error (%)")
plt.legend()

# Plot 3: Sign-Based Correct/Failed Predictions
# A prediction is considered "correct" if the sign (positive/negative/zero) of predicted and true changes match.
pred_sign = torch.sign(predicted_pct[:num_plot])
true_sign = torch.sign(true_pct[:num_plot])
# correct if signs match
correct = (pred_sign == true_sign)
# Convert boolean mask to integer: 1 for correct, 0 for failed.
correct_int = correct.int().numpy()

# Calculate error rate for sign predictions
error_rate = 100 * (1 - np.mean(correct_int))
plt.subplot(3, 1, 3)
# For visualization, plot correct predictions as green dots and failed predictions as red dots.
for i in range(num_plot):
    if correct_int[i] == 1:
        plt.scatter(i, 0, color="green", s=50)  # correct prediction
    else:
        plt.scatter(i, 0, color="red", s=50)      # failed prediction

plt.axhline(0, color="black", linestyle="--")
plt.title(f"Sign-Based Prediction: Error Rate = {error_rate:.2f}% (Green: Correct, Red: Failed)")
plt.xlabel("Sample Index")
plt.yticks([])  # Hide y-axis ticks for clarity

plt.tight_layout()
plt.show()
