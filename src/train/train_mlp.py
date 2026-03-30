import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLPSeasonalBaseline


def _auroc_from_errors(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb    = xb.to(device)
            err   = model.raw_errors(xb)
            score = err.max(dim=1).values
            all_scores.append(score.cpu().numpy())
            all_labels.append(yb.numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    return float(roc_auc_score(labels, scores))


def train_mlp(X_train, y_train,
              X_val,   y_val,
              cfg, out_path,
              X_test=None, y_test=None):
    """
    Train the MLP seasonal baseline.

    Identical objective to GRUSeasonalBaseline — predict demand from time
    features only — but using a pointwise MLP instead of a recurrent network.
    Each timestep is processed independently (no hidden state), which is the
    correct architecture when input features carry no sequential dependency.

    Parameters
    ----------
    X_train / X_val / X_test : (N, W, D)
    y_train / y_val / y_test : (N,)  binary labels
    """
    torch.manual_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[MLP] device={device}")

    time_dim = X_train.shape[-1] - 1   # 4

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val,   dtype=torch.float32)
    yva = torch.tensor(y_val,   dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr),
                           batch_size=cfg.MLP_BATCH, shuffle=True,  drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva, yva),
                           batch_size=cfg.MLP_BATCH, shuffle=False, drop_last=False)

    has_test = X_test is not None and y_test is not None
    if has_test:
        Xte = torch.tensor(X_test, dtype=torch.float32)
        yte = torch.tensor(y_test, dtype=torch.float32)
        te_loader = DataLoader(TensorDataset(Xte, yte),
                               batch_size=cfg.MLP_BATCH, shuffle=False, drop_last=False)

    print(f"[MLP] train windows: {len(X_train)} (normal-only)  time_dim={time_dim}")

    model = MLPSeasonalBaseline(
        time_dim = time_dim,
        hidden_1 = cfg.MLP_HIDDEN_1,
        hidden_2 = cfg.MLP_HIDDEN_2,
        dropout  = cfg.MLP_DROPOUT,
    ).to(device)

    opt       = torch.optim.Adam(model.parameters(), lr=cfg.MLP_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=cfg.MLP_LR_FACTOR, patience=cfg.MLP_LR_PATIENCE,
    )

    best_val_auroc   = -1.0
    best_test_auroc  = None
    patience_counter = 0

    sep    = "─" * 95
    header = (f"{'Ep':>4} {'tr_MSE':>10} {'va_MSE':>10} "
              f"{'val_AUROC':>10} {'test_AUROC':>11} {'lr':>10}")
    print(f"\n[MLP] Seasonal baseline training  (epochs={cfg.MLP_EPOCHS}, "
          f"patience={cfg.MLP_PATIENCE})")
    print(sep); print(header); print(sep)

    for epoch in range(cfg.MLP_EPOCHS):
        model.train()
        tr_mse, n = 0.0, 0
        for xb, _ in tr_loader:
            xb     = xb.to(device)
            x_time = xb[:, :, 1:]
            x_val  = xb[:, :, 0]
            opt.zero_grad()
            pred   = model(x_time)
            loss   = F.mse_loss(pred, x_val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MLP_GRAD_CLIP)
            opt.step()
            tr_mse += loss.item() * xb.size(0)
            n      += xb.size(0)
        tr_mse /= max(n, 1)

        model.eval()
        va_mse, n = 0.0, 0
        with torch.no_grad():
            for xb, _ in va_loader:
                xb     = xb.to(device)
                x_time = xb[:, :, 1:]
                x_val2 = xb[:, :, 0]
                pred   = model(x_time)
                va_mse += F.mse_loss(pred, x_val2).item() * xb.size(0)
                n      += xb.size(0)
        va_mse /= max(n, 1)

        val_auroc  = _auroc_from_errors(model, va_loader,  device)
        test_auroc = _auroc_from_errors(model, te_loader,  device) if has_test else float("nan")

        scheduler.step(val_auroc)
        cur_lr = opt.param_groups[0]["lr"]

        marker = " ◀ best" if val_auroc > best_val_auroc else ""
        print(f"[MLP] {epoch+1:4d}/{cfg.MLP_EPOCHS}  "
              f"tr_MSE={tr_mse:.5f}  va_MSE={va_mse:.5f}  "
              f"val_AUROC={val_auroc:.4f}  test_AUROC={test_auroc:.4f}  "
              f"lr={cur_lr:.2e}{marker}")

        if val_auroc > best_val_auroc:
            best_val_auroc   = val_auroc
            best_test_auroc  = test_auroc
            patience_counter = 0
            torch.save(model.state_dict(), out_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.MLP_PATIENCE:
                print(sep)
                print(f"[MLP] Early stop at epoch {epoch+1}  "
                      f"best val_AUROC={best_val_auroc:.4f}  "
                      f"test_AUROC@best={best_test_auroc:.4f}")
                break

    print(sep)
    print(f"[MLP] Done. Best val_AUROC={best_val_auroc:.4f}  test_AUROC@best={best_test_auroc}")

    # ── fit error normalisation ───────────────────────────────────────────
    model.load_state_dict(torch.load(out_path, map_location=device))
    model.eval()
    all_errs = []
    with torch.no_grad():
        for xb, _ in DataLoader(TensorDataset(Xtr, ytr),
                                 batch_size=512, shuffle=False):
            xb = xb.to(device)
            all_errs.append(model.raw_errors(xb).cpu())
    flat = torch.cat(all_errs, dim=0).reshape(-1)
    model.err_mean.copy_(flat.mean())
    model.err_std.copy_(flat.std())
    print(f"[MLP] err_mean={model.err_mean.item():.5f}  err_std={model.err_std.item():.5f}")
    torch.save(model.state_dict(), out_path)
    print(f"[MLP] Saved with error stats: {out_path}")
    return out_path
