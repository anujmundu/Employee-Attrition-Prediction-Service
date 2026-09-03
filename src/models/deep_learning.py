import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.core.config import AUTOENCODER_PATH, VAE_PATH, TABULAR_RESNET_PATH


# -------------------------------------------------------------
# 1. Deep Autoencoder (Reconstruction Anomaly Detector)
# -------------------------------------------------------------
class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 24),
            nn.LeakyReLU(0.1),
            nn.Linear(24, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 24),
            nn.LeakyReLU(0.1),
            nn.Linear(24, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# -------------------------------------------------------------
# 2. Variational Autoencoder (VAE - Latent Manifold Anomaly)
# -------------------------------------------------------------
class VAENet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 6):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(48, latent_dim)
        self.fc_logvar = nn.Linear(48, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.ReLU(),
            nn.Linear(48, input_dim),
        )

    def encode(self, x):
        h = self.fc_in(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


# -------------------------------------------------------------
# 3. Tabular ResNet (Deep Classification with Skip Connections)
# -------------------------------------------------------------
class TabularResNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        
        # ResBlock 1
        self.block1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
        )
        
        # ResBlock 2
        self.block2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
        )
        self.proj = nn.Linear(64, 32)
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = torch.relu(self.input_layer(x))
        h = torch.relu(h + self.block1(h))
        h = torch.relu(self.proj(h) + self.block2(h))
        return self.head(h)


# -------------------------------------------------------------
# Training Orchestrator
# -------------------------------------------------------------
def train_deep_models(X_processed: np.ndarray, y: np.ndarray = None, epochs: int = 15):
    """Trains Autoencoder, VAE, and Tabular ResNet on preprocessed data."""
    input_dim = X_processed.shape[1]
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 1. Train Autoencoder
    print("Training PyTorch Deep Autoencoder...")
    ae = AutoencoderNet(input_dim)
    opt_ae = torch.optim.AdamW(ae.parameters(), lr=0.002, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    ae.train()
    for _ in range(epochs):
        for (batch,) in loader:
            recon = ae(batch)
            loss = loss_fn(recon, batch)
            opt_ae.zero_grad()
            loss.backward()
            opt_ae.step()
    torch.save(ae.state_dict(), AUTOENCODER_PATH)
    print(f"Autoencoder saved -> {AUTOENCODER_PATH}")
    
    # 2. Train VAE
    print("Training PyTorch Variational Autoencoder (VAE)...")
    vae = VAENet(input_dim)
    opt_vae = torch.optim.Adam(vae.parameters(), lr=0.002)
    
    vae.train()
    for _ in range(epochs):
        for (batch,) in loader:
            recon, mu, logvar = vae(batch)
            recon_loss = nn.functional.mse_loss(recon, batch, reduction="sum")
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + 0.05 * kld
            opt_vae.zero_grad()
            total_loss.backward()
            opt_vae.step()
    torch.save(vae.state_dict(), VAE_PATH)
    print(f"VAE saved -> {VAE_PATH}")
    
    # 3. Train Tabular ResNet
    if y is not None:
        print("Training PyTorch Tabular ResNet...")
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        res_dataset = TensorDataset(X_tensor, y_tensor)
        res_loader = DataLoader(res_dataset, batch_size=32, shuffle=True)
        
        resnet = TabularResNet(input_dim)
        opt_res = torch.optim.AdamW(resnet.parameters(), lr=0.001)
        bce_loss = nn.BCELoss()
        
        resnet.train()
        for _ in range(epochs):
            for bx, by in res_loader:
                pred = resnet(bx)
                loss = bce_loss(pred, by)
                opt_res.zero_grad()
                loss.backward()
                opt_res.step()
        torch.save(resnet.state_dict(), TABULAR_RESNET_PATH)
        print(f"Tabular ResNet saved -> {TABULAR_RESNET_PATH}")
