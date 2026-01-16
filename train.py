import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import joblib
import json
from preprocessing import load_data, scale_data
from model import Autoencoder
from metrics import compute_spe, compute_threshold
import matplotlib.pyplot as plt
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_PATH = config["paths"]["train_data"]
ARTIFACTS_DIR = config["paths"]["artifacts_dir"]

BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["learning_rate"]

INPUT_DIM = config["model"]["input_dim"]
LATENT_DIM = config["model"]["latent_dim"]
HIDDEN_LAYERS = config["model"]["hidden_layers"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1. Load & preprocess training data
# -----------------------------
df = load_data(TRAIN_PATH)
X_train, scaler = scale_data(df, fit=True)



# Save scaler
joblib.dump(scaler, ARTIFACTS_DIR + "scaler.npz")

# Convert to PyTorch tensor
X_tensor = torch.tensor(X_train, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# 2. Initialize autoencoder
# -----------------------------
model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_layers=HIDDEN_LAYERS)
model.to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# 3. Training loop with loss tracking
# -----------------------------
train_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for batch in loader:
        x_batch = batch[0].to(DEVICE)
        optimizer.zero_grad()
        x_hat = model(x_batch)
        loss = criterion(x_hat, x_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
    epoch_loss /= len(loader.dataset)
    train_losses.append(epoch_loss)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {epoch_loss:.6f}")

# -----------------------------
# 4. Compute SPE on training set
# -----------------------------
model.eval()
with torch.no_grad():
    X_tensor_device = X_tensor.to(DEVICE)
    X_recon = model(X_tensor_device).cpu().numpy()

spe = compute_spe(X_train, X_recon)
spe_threshold = compute_threshold(spe, multiplier=3)

# Save SPE threshold
with open(ARTIFACTS_DIR + "spe_limits.json", "w") as f:
    json.dump({"spe_threshold": spe_threshold.tolist()}, f)

# -----------------------------
# 5. Save model
# -----------------------------
torch.save(model.state_dict(), ARTIFACTS_DIR + "autoencoder.pt")
print("Training complete. Artifacts saved: scaler, autoencoder, SPE threshold")

# -----------------------------
# 6. PERFORMANCE MONITORING
# -----------------------------

# 6a. Training loss over epochs
plt.figure("Training Loss")
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# 6b. Sample reconstructions
sample_idx = np.random.choice(len(X_train), 5, replace=False)
X_sample = torch.tensor(X_train[sample_idx], dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    X_recon_sample = model(X_sample).cpu().numpy()

for i, row in enumerate(X_sample.cpu().numpy()):
    plt.figure(f"Sample {i} Reconstruction")
    plt.plot(row, label="Original")
    plt.plot(X_recon_sample[i], label="Reconstruction")
    plt.title(f"Sample {i} Original vs Reconstructed")
    plt.legend()
plt.show()

# 6c. SPE distribution
plt.figure("SPE Distribution")
plt.hist(spe, bins=50, color="skyblue", edgecolor="k")
plt.axvline(spe_threshold, color='r', linestyle='--', label="SPE Threshold")
plt.title("SPE Distribution on Training Data")
plt.xlabel("SPE")
plt.ylabel("Frequency")
plt.legend()
plt.show()
