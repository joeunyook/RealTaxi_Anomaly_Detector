import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.macro_gru import MacroGRU


def train_macro_gru(micro_va: np.ndarray,
                    y_val:    np.ndarray,
                    cfg,
                    out_path: str,
                    device:   torch.device) -> str:
    """
    Train the MacroGRU on validation micro-score sequences.

    Parameters
    ----------
    micro_va : (N_val, 3)  — per-slot [MLP_score, KRN_score, LOF_score]
    y_val    : (N_val,)    — binary slot-level labels
    cfg      : MacroCfg
    out_path : where to save the best model weights
    device   : torch device

    Returns
    -------
    out_path : str
    """
    W = cfg.WINDOW
    N = len(micro_va)

    # Build 6-step sliding sequences from val micro scores
    seqs   = np.stack([micro_va[i - W + 1 : i + 1] for i in range(W - 1, N)])
    labels = y_val[W - 1:].astype(np.float32)   # label of the last slot in each seq

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"[MACRO] Training on {len(seqs)} val sequences  "
          f"(pos={n_pos}, neg={n_neg})  "
          f"epochs={cfg.EPOCHS}  patience={cfg.PATIENCE}")

    X_t = torch.tensor(seqs,   dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=cfg.BATCH, shuffle=True, drop_last=False)

    model = MacroGRU(input_dim=3, hidden=cfg.HIDDEN, layers=cfg.LAYERS).to(device)

    # Class-balanced positive weight to handle normal >> anomaly imbalance
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt        = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    best_loss        = float("inf")
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit  = model(xb)
            loss   = criterion(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n          += xb.size(0)

        avg_loss = total_loss / max(n, 1)

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), out_path)
            marker = " ◀ best"
        else:
            patience_counter += 1
            marker = ""

        if (epoch + 1) % 10 == 0 or marker:
            print(f"[MACRO] ep {epoch+1:3d}/{cfg.EPOCHS}  loss={avg_loss:.5f}{marker}")

        if patience_counter >= cfg.PATIENCE:
            print(f"[MACRO] Early stop at epoch {epoch + 1}  best_loss={best_loss:.5f}")
            break

    print(f"[MACRO] Done. Best loss={best_loss:.5f}  Saved: {out_path}")
    return out_path
