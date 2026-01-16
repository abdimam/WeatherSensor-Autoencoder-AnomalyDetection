import numpy as np

def compute_spe(X, X_hat):
    residual = X - X_hat
    spe = np.sum(residual**2, axis=1)
    return spe

def compute_threshold(spe, multiplier=3):
    mean = np.mean(spe)
    std = np.std(spe)
    return mean + multiplier*std
