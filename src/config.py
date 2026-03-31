# Here we put hyperparameters, paths, and experimen constants
# Hyperparameters : LofCfg, TrainCfg, DataCfg
# Labeling  : Point-level — window label = raw label of its last timestep (METER-style, no horizon inflation)
# Split     : METER-style — train = normal-only windows from first 60% of timeline;
#             val/test = all windows in next 20% / final 20% (preserves realistic anomaly rate)
# Inductive bias : daily pattern of morning/evening taxi demand taken into account, we added sin/cos time features to encode cyclic preiodicity

# Hyperparameters are currently fixed. Future iterations will use Vertex AI hyperparameter tuning to automatically search the parameter space and optimize validation performance

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_CSV: Path = ROOT / "data" / "nyc_taxi.csv"
    OUT_DIR: Path = ROOT / "outputs"
    FIG_DIR: Path = OUT_DIR / "figures"
    TAB_DIR: Path = OUT_DIR / "tables"
    MODEL_DIR: Path = ROOT / "models"

@dataclass(frozen=True)
class DataCfg:

    WINDOW: int = 2                # 2 × 30min = 1h micro window
    #   sin/cos encoding gives every slot full positional identity, so a 24h
    #   window adds no context — only dilutes the onset signal.  1h is enough
    #   to give the MLP a transition step while staying effectively pointwise.
    STRIDE: int = 1
    # chronological splits
    TRAIN_FRAC: float = 0.60
    VAL_FRAC: float = 0.20         # test = rest
    # feature options
    USE_TIME_FEATURES: bool = True # sin/cos hour + day-of-week


@dataclass(frozen=True)
class MacroCfg:
    """Second-stage GRU that consumes 3h of per-slot micro scores."""
    WINDOW:   int   = 48       # 48 × 30min = 24h context window for macro GRU
    HIDDEN:   int   = 32
    LAYERS:   int   = 1
    EPOCHS:   int   = 50
    LR:       float = 1e-3
    PATIENCE: int   = 10
    BATCH:    int   = 128

    # F-beta for threshold selection
    MICRO_FBETA: float = 2.0   # τ_micro: recall 4× > precision (micro FAs are free)
    MACRO_FBETA: float = 1.5   # τ_macro: recall 2.25× > precision (GRU already filters spikes)

@dataclass(frozen=True)
class TrainCfg:
    SEED: int = 42
    DEVICE: str = "cuda"

    # ── RNN ──────────────────────────────────────────────────────────────
    RNN_EPOCHS: int = 100
    RNN_LR: float = 1e-3
    RNN_HIDDEN: int = 128          # was 64
    RNN_LAYERS: int = 2            # was 1
    RNN_BATCH: int = 256
    RNN_DROPOUT: float = 0.3       # was 0.0
    RNN_GRAD_CLIP: float = 1.0     # prevents exploding gradients
    RNN_PATIENCE: int = 15         # early-stop on val AUROC

    # LR schedule: ReduceLROnPlateau (maximise AUROC)
    RNN_LR_FACTOR: float = 0.5
    RNN_LR_PATIENCE: int = 7

    # ── MLP ──────────────────────────────────────────────────────────────
    MLP_EPOCHS: int = 100
    MLP_LR: float = 1e-3
    MLP_HIDDEN_1: int = 128
    MLP_HIDDEN_2: int = 64
    MLP_DROPOUT: float = 0.2
    MLP_BATCH: int = 256
    MLP_PATIENCE: int = 15
    MLP_GRAD_CLIP: float = 1.0
    MLP_LR_FACTOR: float = 0.5
    MLP_LR_PATIENCE: int = 7

    # ── SARIMA ───────────────────────────────────────────────────────────
    SARIMA_P: int = 1    # AR order
    SARIMA_D: int = 0    # differencing
    SARIMA_Q: int = 1    # MA order
    SARIMA_SP: int = 1   # seasonal AR
    SARIMA_SD: int = 0   # seasonal differencing
    SARIMA_SQ: int = 1   # seasonal MA
    SARIMA_S: int = 48   # seasonal period (48 × 30 min = 24 h)


@dataclass(frozen=True)
class LofCfg:
    N_NEIGHBORS: int = 35
    CONTAMINATION: float = 0.1     # used only for internal LOF behavior

@dataclass(frozen=True)
class KernelCfg:
    BANDWIDTH_HOUR: float = 0.08   # Gaussian σ for (sin_h, cos_h) kernel ≈ ±1 slot
    BANDWIDTH_DOW:  float = 0.40   # Gaussian σ for (sin_d, cos_d) kernel ≈ adjacent days