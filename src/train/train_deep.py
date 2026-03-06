"""
Generic training loop for deep temporal anomaly classifiers.

Supports: LSTM, CNN, Transformer (and any nn.Module that accepts
(B, W, D) input and returns (B,) logits for binary classification).

Training details
----------------
- Loss: Binary cross-entropy with logits (BCEWithLogitsLoss)
- Optimizer: Adam
- Early stopping: save best checkpoint based on validation loss
- Class imbalance: positive-weight upweighting proportional to
  the negative/positive ratio in the training split
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np


def _pos_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute positive class weight to handle class imbalance."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(float(n_neg) / float(n_pos), device=device)


def train_deep_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    out_path: str,
    device: torch.device,
) -> str:
    """
    Train a binary classification model and save the best checkpoint.

    Parameters
    ----------
    model : nn.Module
        Instantiated model (already moved to ``device``).
    X_train, X_val : np.ndarray of shape (N, W, D)
    y_train, y_val : np.ndarray of shape (N,) with values in {0, 1}
    epochs : int
    lr : float
    batch_size : int
    out_path : str
        File path to save the best model state dict.
    device : torch.device

    Returns
    -------
    out_path : str
    """
    pw = _pos_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
        val_loss /= max(n, 1)
        print(f"  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path)

    return out_path
