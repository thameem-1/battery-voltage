import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib


def stratified_split(graphs, df_valid):
    df_valid = df_valid.reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        df_valid.index, test_size=0.1, stratify=df_valid["working_ion"], random_state=42
    )
    return [graphs[i] for i in train_idx], [graphs[i] for i in test_idx]


def dual_collate(batch):
    g1_list, g2_list, y_list = zip(*batch)
    g1_batch = Batch.from_data_list(g1_list)
    g2_batch = Batch.from_data_list(g2_list)
    y_batch = torch.as_tensor(y_list, dtype=torch.float32)
    return g1_batch, g2_batch, y_batch


class BaseGNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, g1, g2):
        x1 = self.encode(g1.x, g1.edge_index, g1.edge_attr, g1.batch)
        x2 = self.encode(g2.x, g2.edge_index, g2.edge_attr, g2.batch)
        return self.mlp(x1 + x2).squeeze(1)


class TransformerGNN(BaseGNN):
    def __init__(self, in_dim, hidden_dim, edge_dim):
        super().__init__(hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.encoder = nn.ModuleList([
            TransformerConv(in_dim, hidden_dim, heads=4, concat=True, edge_dim=hidden_dim),
            TransformerConv(4 * hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=hidden_dim),
            TransformerConv(4 * hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=hidden_dim),
            TransformerConv(4 * hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=hidden_dim),
        ])

    def encode(self, x, edge_index, edge_attr, batch):
        edge_attr = self.edge_proj(edge_attr)
        for conv in self.encoder:
            x = F.relu(conv(x, edge_index, edge_attr))
        return self.pool(x, batch)


def train_dual(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for g1, g2, y in loader:
        g1, g2, y = g1.to(device), g2.to(device), y.to(device).view(-1)
        optimizer.zero_grad()
        pred = model(g1, g2)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.numel()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test_dual(model, loader, device):
    model.eval()
    preds, trues = [], []
    for g1, g2, y in loader:
        g1, g2 = g1.to(device), g2.to(device)
        out = model(g1, g2)
        preds.append(out.cpu())
        trues.append(y.view(-1))
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    return (
        mean_absolute_error(trues, preds),
        r2_score(trues, preds),
        mean_squared_error(trues, preds),
        preds,
        trues,
    )


graphs, df_valid = joblib.load("/content/dual_graph_dataset_.pkl")
train_graphs, test_graphs = stratified_split(graphs, df_valid)

train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True, collate_fn=dual_collate)
test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False, collate_fn=dual_collate)

in_dim = train_graphs[0][0].x.shape[1]
edge_dim = train_graphs[0][0].edge_attr.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerGNN(in_dim=in_dim, hidden_dim=128, edge_dim=edge_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 101):
    train_mse = train_dual(model, train_loader, optimizer, device)
    mae, r2, mse, _, _ = test_dual(model, test_loader, device)
    print(f"Epoch {epoch:03d} | Train MSE: {train_mse:.4f} | Test MSE: {mse:.4f} | Test MAE: {mae:.4f} | Test R²: {r2:.4f}")

mae_tr, r2_tr, mse_tr, train_preds, train_targets = test_dual(model, train_loader, device)
mae_te, r2_te, mse_te, test_preds, test_targets = test_dual(model, test_loader, device)

print("\nFinal Test Metrics:")
print(f"R²  = {r2_te:.4f}")
print(f"MSE = {mse_te:.4f}")
print(f"MAE = {mae_te:.4f}")

plt.figure(figsize=(7, 6), dpi=300)
plt.scatter(train_targets, train_preds, label="Train", alpha=0.6, color="blue")
plt.scatter(test_targets, test_preds, label="Test", alpha=0.6, color="red")
min_val = min(train_targets.min(), test_targets.min())
max_val = max(train_targets.max(), test_targets.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
plt.title("TransformerGNN — Actual vs Predicted Voltage", fontsize=14)
plt.xlabel("Actual Voltage (V)", fontsize=12)
plt.ylabel("Predicted Voltage (V)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
