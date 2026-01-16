import pandas as pd
import numpy as np
import torch
import joblib
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from preprocessing import load_data, scale_data
from model import Autoencoder
from metrics import compute_spe

# -----------------------------
# ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser(description="Run process monitoring on a dataset")
parser.add_argument("--data_path", type=str, required=True, help="Path to CSV for monitoring")
parser.add_argument("--artifacts_dir", type=str, default="artifacts/", help="Folder where model/scaler/SPE threshold are stored")
args = parser.parse_args()

DATA_PATH = args.data_path
ARTIFACTS_DIR = args.artifacts_dir
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1. Load data
# -----------------------------
df = load_data(DATA_PATH)  # parse_dates inside function

# -----------------------------
# 2. Scale data
# -----------------------------
scaler = joblib.load(ARTIFACTS_DIR + "scaler.npz")
X, _ = scale_data(df, scaler=scaler, fit=False)  # always returns two values

X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

# -----------------------------
# 3. Load autoencoder
# -----------------------------
INPUT_DIM = X.shape[1]
LATENT_DIM = 8
HIDDEN_LAYERS = [16]

model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_layers=HIDDEN_LAYERS)
model.load_state_dict(torch.load(ARTIFACTS_DIR + "autoencoder.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# 4. Reconstruct and compute SPE
# -----------------------------
with torch.no_grad():
    X_recon = model(X_tensor).cpu().numpy()

spe = compute_spe(X, X_recon)

# -----------------------------
# 5. Load threshold and flag anomalies
# -----------------------------
with open(ARTIFACTS_DIR + "spe_limits.json") as f:
    spe_threshold = json.load(f)["spe_threshold"]

anomalies = spe > spe_threshold

df_result = df.copy()
df_result["SPE"] = spe
df_result["anomaly"] = anomalies

# Save monitored CSV
output_path = DATA_PATH.replace(".csv", "_monitored.csv")
df_result.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
print(f"Number of anomalies: {anomalies.sum()} / {len(anomalies)}")

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PREPARE DATA
# -----------------------------
df_result["date"] = pd.to_datetime(df_result["date"])
df_result.set_index("date", inplace=True)

anomaly_matrix = df_result.drop(columns=["SPE"]).copy()
anomaly_matrix = anomaly_matrix.where(df_result["anomaly"], np.nan)
anomaly_matrix = anomaly_matrix.apply(pd.to_numeric, errors="coerce")

# -----------------------------
# 1️⃣ SPE over time
# -----------------------------
plt.figure("SPE Plot")
plt.plot(df_result.index, df_result["SPE"], label="SPE")
plt.axhline(y=spe_threshold, color='r', linestyle='--', label="SPE Threshold")
plt.scatter(df_result.index[df_result["anomaly"]],
            df_result["SPE"][df_result["anomaly"]],
            color='red', label="Anomalies")
plt.title("SPE over Time with Anomalies")
plt.xlabel("Time")
plt.ylabel("SPE")
plt.legend()
plt.tight_layout()


# -----------------------------
# 2️⃣ Daily anomaly counts
# -----------------------------
plt.figure("Daily Anomalies")
daily_anomalies = df_result["anomaly"].resample("D").sum()
plt.bar(daily_anomalies.index, daily_anomalies.values, color="tomato")
plt.title("Daily Anomalies Count")
plt.xlabel("Date")
plt.ylabel("Number of Anomalies")
plt.tight_layout()


# -----------------------------
# 3️⃣ Heatmap of anomalies
# -----------------------------
plt.figure("Heatmap of Anomalies")
sns.heatmap(anomaly_matrix.T, cmap="Reds", cbar=False)
plt.title("Heatmap of Anomalies Across Variables")
plt.xlabel("Time")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()  # BLOCKING: stays open independently

print("All plots are ready. Close all windows to finish.")
