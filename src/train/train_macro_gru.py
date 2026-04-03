import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.macro_gru import MacroGRU


def _build_sample_weights(labels: np.ndarray,
                           n_pos: int, n_neg: int,
                           post_event_slots: int,
                           post_event_weight: float) -> np.ndarray:
    """
    Per-sample loss weights encoding two inductive biases:

    1. Class balance  — positives get weight n_neg/n_pos so rare anomaly
                        slots are not drowned out by the majority normal class.

    2. Post-event hard negatives — the GRU hidden state stays saturated for
       several hours after a real event ends, causing ghost false alarms.
       The first `post_event_slots` negative slots after each event end get
       weight `post_event_weight` so the model learns to cool down quickly.
       These weights override class-balance for those specific slots.
    """
    pos_w = float(n_neg) / max(n_pos, 1)
    w = np.where(labels == 1, pos_w, 1.0).astype(np.float32)

    # mark post-event negatives
    in_event   = False
    countdown  = 0
    for i, lbl in enumerate(labels):
        if lbl == 1:
            in_event  = True
            countdown = 0
        else:
            if in_event:
                in_event  = False
                countdown = post_event_slots
            if countdown > 0:
                w[i]      = post_event_weight
                countdown -= 1

    return w


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
    """
    W = cfg.WINDOW
    N = len(micro_va)

    seqs   = np.stack([micro_va[i - W + 1 : i + 1] for i in range(W - 1, N)])
    labels = y_val[W - 1:].astype(np.float32)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    sample_w = _build_sample_weights(
        labels, n_pos, n_neg,
        cfg.POST_EVENT_SLOTS, cfg.POST_EVENT_WEIGHT,
    )
    n_post = int((sample_w == cfg.POST_EVENT_WEIGHT).sum())
    print(f"[MACRO] Training on {len(seqs)} val sequences  "
          f"(pos={n_pos}, neg={n_neg}, post-event hard-neg={n_post})  "
          f"epochs={cfg.EPOCHS}  patience={cfg.PATIENCE}")

    X_t = torch.tensor(seqs,     dtype=torch.float32)
    y_t = torch.tensor(labels,   dtype=torch.float32)
    w_t = torch.tensor(sample_w, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_t, y_t, w_t),
                        batch_size=cfg.BATCH, shuffle=True, drop_last=False)

    model     = MacroGRU(input_dim=cfg.INPUT_DIM, hidden=cfg.HIDDEN, layers=cfg.LAYERS).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    opt       = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    best_loss        = float("inf")
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            logit = model(xb)
            loss  = (criterion(logit, yb) * wb).mean()
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
