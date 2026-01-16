import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers=[16]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        
        # decoder
        layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_layers):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

