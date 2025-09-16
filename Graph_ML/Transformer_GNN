
import numpy as np, pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def stratified_split(graphs, df_valid):
    df_valid = df_valid.reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        df_valid.index, test_size=0.1, stratify=df_valid['working_ion'], random_state=42
    )
    return [graphs[i] for i in train_idx], [graphs[i] for i in test_idx]


class BaseGNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, g1, g2):
        x1 = self.encode(g1.x, g1.edge_index, getattr(g1, 'edge_attr', None), g1.batch)
        x2 = self.encode(g2.x, g2.edge_index, getattr(g2, 'edge_attr', None), g2.batch)
        return self.mlp(x1 + x2).squeeze(1)

# model class - change this part for training GCN, GAT, GATv2

class TransformerGNN(BaseGNN):
    def __init__(self, in_dim, hidden_dim, edge_dim=41):
        super().__init__(hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.encoder = nn.ModuleList([
            TransformerConv(in_dim, hidden_dim, heads=4, concat=True, edge_dim=hidden_dim),
            TransformerConv(4 * hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=hidden_dim),
            TransformerConv(4 * hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=hidden_dim),
            TransformerConv(4 * hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=hidden_dim)
        ])

    def encode(self, x, edge_index, edge_attr, batch):
        edge_attr = self.edge_proj(edge_attr)
        for conv in self.encoder:
            x = F.relu(conv(x, edge_index, edge_attr))
        return self.pool(x, batch)


# Train & Evaluate
def train_dual(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for g1, g2, y in loader:
        g1, g2, y = g1.to(device, non_blocking=True), g2.to(device, non_blocking=True), y.to(device).view(-1)
        optimizer.zero_grad()
        pred = model(g1, g2)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test_dual(model, loader, device):
    model.eval()
    preds, trues = [], []
    for g1, g2, y in loader:
        g1, g2 = g1.to(device, non_blocking=True), g2.to(device, non_blocking=True)
        preds.append(model(g1, g2).cpu())
        trues.append(y.view(-1))
    return (
        mean_absolute_error(torch.cat(trues), torch.cat(preds)),
        r2_score(torch.cat(trues), torch.cat(preds)),
        torch.cat(preds).numpy(),
        torch.cat(trues).numpy()
    )

import joblib

# Load graph data
graphs, df_valid = joblib.load("/content/dual_graph_dataset_.pkl")

train_graphs, test_graphs = stratified_split(graphs, df_valid)
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False, pin_memory=True)

# Ensure x is 2D
for g_list in [train_graphs, test_graphs]:
    for i in range(len(g_list)):
        for g in [g_list[i][0], g_list[i][1]]:
            if g.x.dim() == 1:
                g.x = g.x.view(-1, 1)

# Final training
in_dim = train_graphs[0][0].x.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerGNN(in_dim=in_dim, hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 101):
    loss = train_dual(model, train_loader, optimizer, device)
    print(f"Epoch {epoch:02d}: Train Loss = {loss:.4f}")

mae, r2, preds, trues = test_dual(model, test_loader, device)
print(f"Test MAE = {mae:.4f} V | R² = {r2:.4f}")

# Plot results
plt.figure(figsize=(6, 6))
plt.scatter(trues, preds, alpha=0.6)
plt.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--')
plt.xlabel("Actual Voltage (V)")
plt.ylabel("Predicted Voltage (V)")
plt.title("Dual-GNN Voltage Prediction")
plt.grid(True)
plt.tight_layout()
plt.show()
