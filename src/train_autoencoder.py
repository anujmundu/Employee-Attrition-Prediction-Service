import joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PIPELINE_PATH = "models/model_v1.joblib"
DATA_PATH = "data/raw.csv"
ENCODER_PATH = "models/autoencoder.pt"


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def main():
    # Load preprocessing from supervised pipeline
    pipeline = joblib.load(PIPELINE_PATH)
    preprocessor = pipeline.named_steps["preprocessing"]

    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Attrition"])
    X_processed = preprocessor.transform(X)

    # Convert to torch tensors
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Autoencoder(X_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Train
    for epoch in range(20):
        total_loss = 0
        for (batch,) in loader:
            recon = model(batch)
            loss = loss_fn(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss {total_loss:.4f}")

    torch.save(model.state_dict(), ENCODER_PATH)
    print("Autoencoder saved.")


if __name__ == "__main__":
    main()
