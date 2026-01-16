# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Load CSV and parse date column"""
    df = pd.read_csv(path, parse_dates=["date"])
    df_clean = df.fillna(df.mean())
# or for median:
# df_clean = df.fillna(df.median())

    return df_clean

def scale_data(df, scaler=None, fit=False):
    features = df.drop(columns=["date"])
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
    else:
        X_scaled = scaler.transform(features)
    return X_scaled, scaler

