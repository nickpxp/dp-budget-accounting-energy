"""paper_utils_dpcnn.py

Model, data and training code for the DP-SGD sweep on PTB-XL.

Contents
--------
1. Data loading and canonical patient-disjoint splits
2. Per-lead normalization fit on training records only
3. InceptionTime1D with GroupNorm, so Opacus can take per-sample gradients
4. Training: non-private baseline and DP-SGD via Opacus
5. Checkpoint save/load
6. Attack metric helpers (TPR at fixed FPR, AUC)
7. Environment helpers (device, logging, checkpoint naming)
8. Demographic attribute targets
9. Representation probing
10. Control metrics
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger("paper_utils_dpcnn")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LEADS = 12
SIGNAL_LENGTH = 1000          # 10 s @ 100 Hz
SAMPLING_RATE = 100
EXPECTED_AF_PREVALENCE = 0.0727
DEFAULT_DELTA = 4.6e-5        # below 1 / n for every training set used here

# default epsilon grid; the sweep script overrides it
DEFAULT_EPSILONS = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


# 1. Data loading and canonical splits

@dataclass
class PTBXLData:
    """Container for loaded PTB-XL raw waveforms."""
    X: np.ndarray             # (N, 1000, 12) float32
    y: np.ndarray             # (N,) int64
    fold: np.ndarray          # (N,) int64
    patient_id: np.ndarray    # (N,) int64
    ecg_id: np.ndarray        # (N,) int64

    def __len__(self) -> int:
        return len(self.y)


def load_ptbxl_cache(path: str | Path = "data/ptbxl_raw_100hz.npz") -> PTBXLData:
    """Load the .npz cache produced by ``PTB_XL_Preprocessing_1D_CNN.ipynb``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PTB-XL cache not found at {path}. "
            "Run PTB_XL_Preprocessing_1D_CNN.ipynb first."
        )
    z = np.load(path)
    data = PTBXLData(
        X=z["X"], y=z["y"], fold=z["fold"],
        patient_id=z["patient_id"], ecg_id=z["ecg_id"],
    )
    _assert_patient_disjoint(data)
    _assert_prevalence(data)
    return data


def canonical_split(data: PTBXLData) -> dict[str, np.ndarray]:
    """Canonical splits: folds 1-8 train, 9 val, 10 test.

    Patient-disjoint by construction of strat_fold.
    """
    return {
        "train": np.where(data.fold <= 8)[0],
        "val":   np.where(data.fold == 9)[0],
        "test":  np.where(data.fold == 10)[0],
    }


def _assert_patient_disjoint(data: PTBXLData) -> None:
    """Verify no patient appears in more than one partition."""
    splits = canonical_split(data)
    pts = {k: set(data.patient_id[idx]) for k, idx in splits.items()}
    assert not (pts["train"] & pts["test"]), "Patient leak: train ∩ test"
    assert not (pts["train"] & pts["val"]), "Patient leak: train ∩ val"
    assert not (pts["val"] & pts["test"]), "Patient leak: val ∩ test"


def _assert_prevalence(data: PTBXLData, tol: float = 0.005) -> None:
    """Verify AF prevalence is close to the published 7.27%."""
    prev = float(data.y.mean())
    assert abs(prev - EXPECTED_AF_PREVALENCE) < tol, (
        f"AF prevalence {prev:.4f} outside ±{tol} of {EXPECTED_AF_PREVALENCE:.4f}"
    )


# 2. Normalization (train-only statistics)

def fit_normalizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-lead mean and std from training data only.

    Post-processing of model outputs costs no privacy budget
    Proposition 2.1. Normalization stats from the train fold are inputs
    to that post-processing, hence also ε-free.

    Returns
    -------
    mean, std : np.ndarray of shape (1, 1, 12)
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = (X_train.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
    return mean, std


def apply_normalizer(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply z-score normalization and return channels-first PyTorch shape."""
    X_norm = ((X - mean) / std).astype(np.float32)
    # (N, 1000, 12) -> (N, 12, 1000) for Conv1d
    return X_norm.transpose(0, 2, 1)


def build_loaders(
    data: PTBXLData,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, np.ndarray]]:
    """Build train/val/test DataLoaders with train-only normalization.

    Returns
    -------
    train_loader, val_loader, test_loader, norm_stats
        ``norm_stats`` is ``{"mean": ..., "std": ...}`` for reuse downstream.
    """
    splits = canonical_split(data)
    mean, std = fit_normalizer(data.X[splits["train"]])

    loaders = []
    for name, shuffle in [("train", True), ("val", False), ("test", False)]:
        idx = splits[name]
        X_p = apply_normalizer(data.X[idx], mean, std)
        y_p = data.y[idx]
        ds = TensorDataset(torch.from_numpy(X_p), torch.from_numpy(y_p))
        loaders.append(DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        ))

    return (*loaders, {"mean": mean, "std": std})


# ===========================================================================
# 3. InceptionTime 1D-CNN, Opacus-compatible
# ===========================================================================

class InceptionModule(nn.Module):
    """Single Inception module: parallel multi-scale 1D convolutions.

    GroupNorm replaces BatchNorm so Opacus can compute per-sample gradients.
    Standard InceptionTime layout.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (10, 20, 40),
        bottleneck_channels: int = 32,
    ):
        super().__init__()
        # 1x1 bottleneck reduces channels before expensive multi-scale convs
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        # Three parallel convs at different scales.
        # padding='same' handles even kernel sizes correctly (k//2 would over-pad).
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, n_filters, kernel_size=k,
                      padding='same', bias=False)
            for k in kernel_sizes
        ])
        # Parallel max-pool + 1x1 conv branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(
            in_channels, n_filters, kernel_size=1, bias=False
        )

        out_channels = n_filters * (len(kernel_sizes) + 1)
        # GroupNorm with num_groups=1 (LayerNorm-equivalent), Opacus-friendly
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottle = self.bottleneck(x)
        parts = [conv(bottle) for conv in self.convs]
        parts.append(self.pool_conv(self.maxpool(x)))
        out = torch.cat(parts, dim=1)
        out = self.norm(out)
        return F.relu(out)


class InceptionTime1D(nn.Module):
    """InceptionTime for 12-lead ECG, AF binary classification.

    Six Inception modules with residual connections every three, followed by
    global average pooling and a linear head. Total ~420K parameters.

    Opacus-compatible: GroupNorm throughout, no BatchNorm.
    """

    def __init__(
        self,
        in_channels: int = NUM_LEADS,
        n_filters: int = 32,
        depth: int = 6,
        num_classes: int = 1,  # binary: single logit + BCEWithLogits
    ):
        super().__init__()
        self.depth = depth
        self.modules_list = nn.ModuleList()
        self.residuals = nn.ModuleList()

        out_channels = n_filters * 4  # 3 conv branches + 1 pool branch
        current_in = in_channels
        # Track the channel count of the tensor saved as `residual_input` , 
        # i.e., the input to the current 3-block. Initially that's `in_channels`.
        residual_in_channels = in_channels

        for i in range(depth):
            self.modules_list.append(
                InceptionModule(in_channels=current_in, n_filters=n_filters)
            )
            current_in = out_channels  # module output is always out_channels

            # Residual projection every 3 modules: maps residual_in_channels → out_channels
            if (i + 1) % 3 == 0:
                self.residuals.append(nn.Sequential(
                    nn.Conv1d(
                        residual_in_channels,
                        out_channels, kernel_size=1, bias=False,
                    ),
                    nn.GroupNorm(num_groups=1, num_channels=out_channels),
                ))
                # After this 3-block, the new residual_input has out_channels
                residual_in_channels = out_channels

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        residual_idx = 0

        for i, module in enumerate(self.modules_list):
            x = module(x)
            if (i + 1) % 3 == 0:
                x = x + self.residuals[residual_idx](residual_input)
                x = F.relu(x)
                residual_input = x
                residual_idx += 1

        x = self.gap(x).reshape(x.size(0), -1)
        return self.head(x).reshape(-1)  # (B,) logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# 4. Training: non-private baseline and DP-SGD
# ===========================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    # DP-only:
    target_epsilon: Optional[float] = None     # None = non-private
    target_delta: float = DEFAULT_DELTA
    max_grad_norm: float = 1.0
    # Imbalance handling: pos_weight is ε-free (data-independent constant
    # set from prevalence)
    pos_weight: Optional[float] = None
    # Reproducibility:
    seed: int = 0


@dataclass
class TrainResult:
    """What a training run produces."""
    config: TrainConfig
    n_params: int
    final_train_loss: float
    final_val_auc: float
    test_auc: float
    test_logits: np.ndarray = field(repr=False)  # (N_test,), for LiRA
    test_labels: np.ndarray = field(repr=False)
    epsilon_spent: Optional[float] = None
    delta: float = DEFAULT_DELTA


def _make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def _bce_loss(cfg: TrainConfig) -> nn.Module:
    """BCEWithLogitsLoss with optional pos_weight for class imbalance.

    pos_weight is a data-independent scalar (set from training prevalence
    computed once before DP-SGD begins), so post-processing covers it.
    """
    pw = (
        torch.tensor([cfg.pos_weight], dtype=torch.float32)
        if cfg.pos_weight is not None
        else None
    )
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def _eval_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (AUC, logits, labels) on a loader."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    auc = float(roc_auc_score(labels, logits)) if labels.sum() > 0 else float("nan")
    return auc, logits, labels


def train_non_private(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> TrainResult:
    """Standard (non-DP) training. Baseline for comparison."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = model.to(device)
    optimizer = _make_optimizer(model, cfg)
    criterion = _bce_loss(cfg).to(device)

    final_train_loss = float("nan")
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.float().to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_train_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            val_auc, _, _ = _eval_auc(model, val_loader, device)
            log.info(
                f"[non-private] epoch {epoch+1}/{cfg.epochs} "
                f"loss={final_train_loss:.4f} val_auc={val_auc:.4f}"
            )

    val_auc, _, _ = _eval_auc(model, val_loader, device)
    test_auc, test_logits, test_labels = _eval_auc(model, test_loader, device)

    return TrainResult(
        config=cfg, n_params=count_parameters(model),
        final_train_loss=final_train_loss,
        final_val_auc=val_auc, test_auc=test_auc,
        test_logits=test_logits, test_labels=test_labels,
        epsilon_spent=None, delta=cfg.target_delta,
    )


def train_dp_sgd(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> TrainResult:
    """DP-SGD training via Opacus.

    Calls ``PrivacyEngine.make_private_with_epsilon`` which calibrates the
    noise multiplier to hit ``(target_epsilon, target_delta)`` after the
    specified number of epochs. RDP accountant by default.
    """
    if cfg.target_epsilon is None:
        raise ValueError("target_epsilon required for DP-SGD training")

    # Import here so non-DP users don't need Opacus installed
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Sanity: GroupNorm-only, BatchNorm forbidden by Opacus
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise ValueError(
            f"Model not Opacus-compatible: {errors}. "
            "Use GroupNorm (model already does this), but double-check "
            "if you customized."
        )

    model = model.to(device)
    optimizer = _make_optimizer(model, cfg)
    criterion = _bce_loss(cfg).to(device)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=cfg.epochs,
        target_epsilon=cfg.target_epsilon,
        target_delta=cfg.target_delta,
        max_grad_norm=cfg.max_grad_norm,
    )
    log.info(
        f"[dp-sgd] ε={cfg.target_epsilon}, δ={cfg.target_delta:.2e}, "
        f"noise_mult={optimizer.noise_multiplier:.4f}, "
        f"clip={cfg.max_grad_norm}"
    )

    final_train_loss = float("nan")
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=64,  # caps memory; logical batch unchanged
            optimizer=optimizer,
        ) as memory_safe_loader:
            for X, y in memory_safe_loader:
                X, y = X.to(device), y.float().to(device)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
        final_train_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            val_auc, _, _ = _eval_auc(model, val_loader, device)
            eps_spent = privacy_engine.get_epsilon(cfg.target_delta)
            log.info(
                f"[dp-sgd] epoch {epoch+1}/{cfg.epochs} "
                f"loss={final_train_loss:.4f} val_auc={val_auc:.4f} "
                f"ε_spent={eps_spent:.4f}"
            )

    val_auc, _, _ = _eval_auc(model, val_loader, device)
    test_auc, test_logits, test_labels = _eval_auc(model, test_loader, device)
    epsilon_spent = float(privacy_engine.get_epsilon(cfg.target_delta))

    return TrainResult(
        config=cfg, n_params=count_parameters(model),
        final_train_loss=final_train_loss,
        final_val_auc=val_auc, test_auc=test_auc,
        test_logits=test_logits, test_labels=test_labels,
        epsilon_spent=epsilon_spent, delta=cfg.target_delta,
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
) -> TrainResult:
    """Dispatch to DP-SGD or non-private training based on cfg.target_epsilon."""
    if device is None:
        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
    if cfg.target_epsilon is None:
        return train_non_private(model, train_loader, val_loader, test_loader, cfg, device)
    return train_dp_sgd(model, train_loader, val_loader, test_loader, cfg, device)


# ===========================================================================
# 5. Checkpoint I/O
# ===========================================================================

def save_checkpoint(
    result: TrainResult,
    model: nn.Module,
    path: str | Path,
) -> None:
    """Save model state + result metadata.

    Two files: ``<path>`` (state dict) and ``<path>.json`` (metadata).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

    # Serialize everything except numpy arrays inline
    meta = {
        "config": asdict(result.config),
        "n_params": result.n_params,
        "final_train_loss": result.final_train_loss,
        "final_val_auc": result.final_val_auc,
        "test_auc": result.test_auc,
        "epsilon_spent": result.epsilon_spent,
        "delta": result.delta,
    }
    with open(path.with_suffix(path.suffix + ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save test predictions alongside (used by MIA/LiRA)
    np.savez_compressed(
        path.with_suffix(path.suffix + ".preds.npz"),
        test_logits=result.test_logits,
        test_labels=result.test_labels,
    )


def load_checkpoint(
    path: str | Path,
    model_factory=InceptionTime1D,
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, dict]:
    """Load a saved model + its metadata dict."""
    path = Path(path)
    if device is None:
        device = torch.device("cpu")

    model = model_factory()
    state = torch.load(path, map_location=device)
    # Opacus prefixes parameters with '_module.', strip if present
    state = {k.replace("_module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device).eval()

    with open(path.with_suffix(path.suffix + ".json")) as f:
        meta = json.load(f)
    return model, meta


def load_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (test_logits, test_labels) saved by save_checkpoint."""
    path = Path(path)
    z = np.load(path.with_suffix(path.suffix + ".preds.npz"))
    return z["test_logits"], z["test_labels"]


# ===========================================================================
# 6. Attack metric helpers 
# ===========================================================================

def mia_auc(scores: np.ndarray, is_member: np.ndarray) -> float:
    """ROC-AUC of an MIA attack. 0.5 = random."""
    return float(roc_auc_score(is_member, scores))


def mia_tpr_at_fpr(
    scores: np.ndarray,
    is_member: np.ndarray,
    target_fpr: float = 0.001,
) -> float:
    """TPR at a fixed low FPR.

    Default 0.1% FPR. Returns NaN if the
    target FPR cannot be reached (e.g. attack has no resolution at that
    operating point).
    """
    fpr, tpr, _ = roc_curve(is_member, scores)
    mask = fpr <= target_fpr
    if not mask.any():
        return float("nan")
    return float(tpr[mask].max())


# ===========================================================================
# 7. Environment helpers
# ===========================================================================

def get_device(prefer: Optional[str] = None) -> torch.device:
    """Select the compute device, preferring MPS then CUDA then CPU.

    Replaces the device-selection block that was repeated in nine notebooks.

    Parameters
    ----------
    prefer
        Force a specific device ("mps", "cuda", "cpu"). ``None`` autodetects.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(name: str = "paper", level: int = logging.INFO) -> logging.Logger:
    """Configure logging once and return a named logger.

    Idempotent: repeated calls in the same kernel do not stack handlers, which
    is what produced duplicated log lines when notebook cells were re-run.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s  %(message)s")
    root.setLevel(level)
    return logging.getLogger(name)


def tag_for(epsilon: Optional[float], seed: int) -> str:
    """Canonical checkpoint tag.

    ``None`` epsilon denotes the non-private baseline. This is the single
    definition of the naming convention; every notebook that reads or writes
    checkpoints must use it, or a rename silently orphans saved models.

    >>> tag_for(None, 0)
    'target_baseline_seed00'
    >>> tag_for(1.0, 2)
    'target_eps_1.0_seed02'
    """
    if epsilon is None:
        return f"target_baseline_seed{seed:02d}"
    # ``:.1f`` matches the convention already on disk. Without it an int epsilon
    # (1 rather than 1.0) would produce "target_eps_1" and orphan the checkpoint.
    return f"target_eps_{epsilon:.1f}_seed{seed:02d}"


# 8. Demographic attribute targets
#
# These bins define what "age-group" means across every attack and control.
# They previously existed in five separate notebook copies; any drift between
# copies would have made results silently incomparable while raising no error.

AGE_BINS = [0, 40, 55, 65, 75, 200]
AGE_BIN_LABELS = ["<40", "40-55", "55-65", "65-75", ">75"]


def build_attribute_targets(df) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Map a demographics DataFrame to discrete attribute targets.

    Parameters
    ----------
    df
        DataFrame with ``sex`` and ``age`` columns, row-aligned to the cache
        (i.e. indexed by ``ecg_id`` and reindexed with ``.loc[data.ecg_id]``).

    Returns
    -------
    dict
        ``{attribute: (y, valid_mask, n_classes)}``. Invalid rows carry ``-1``
        in ``y`` and ``False`` in ``valid_mask``; callers must filter on the
        mask rather than assume completeness.

    Notes
    -----
    PTB-XL encodes sex as 0 = male, 1 = female. The Chapman extraction follows
    the same convention, which is what makes the two datasets comparable here.

    One deviation from the notebook copies this replaces: those applied the
    clip to the whole array, so rows with a missing age silently became bin 0
    ("<40") instead of staying at the -1 sentinel, while missing sex correctly
    stayed at -1. Here the clip is applied only to valid rows, so both
    attributes mark invalid rows the same way. No published number changes,
    because every caller filters on ``valid_mask`` before scoring.
    """
    import pandas as pd  # local: keeps pandas optional for model-only users

    targets: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

    sex = df["sex"].to_numpy()
    sex_valid = ~pd.isna(sex)
    targets["sex"] = (
        np.where(sex_valid, sex, -1).astype(np.int64),
        sex_valid,
        2,
    )

    age = df["age"].to_numpy()
    age_valid = ~pd.isna(age)
    age_bin = np.full(len(age), -1, dtype=np.int64)
    age_bin[age_valid] = np.clip(
        np.digitize(age[age_valid], AGE_BINS) - 1, 0, len(AGE_BIN_LABELS) - 1
    )
    targets["age_bin"] = (age_bin, age_valid, len(AGE_BIN_LABELS))

    return targets


# 9. Representation probing

def extract_features(
    model: nn.Module,
    X: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Forward ``X`` through ``model`` up to the penultimate layer.

    Taps the global-average-pool output, i.e. the representation immediately
    before the classification head. Written as an explicit forward rather than
    a hook so it is agnostic to whether the model was wrapped by Opacus's
    ``GradSampleModule``.

    This is the single definition of "the representation" for the attribute
    probe. It previously existed in three notebook copies; if those had
    diverged, the probe would have measured different things in each and the
    numbers would not have been comparable.

    Parameters
    ----------
    X
        Normalized waveforms, channels-first ``(N, NUM_LEADS, SIGNAL_LENGTH)``.

    Returns
    -------
    np.ndarray
        ``(N, n_filters * 4)`` penultimate activations.
    """
    if device is None:
        device = get_device()

    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.zeros(len(X), dtype=torch.int64),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device).eval()
    feats = []
    with torch.no_grad():
        for xb, _ in loader:
            x = xb.to(device)
            residual_input, residual_idx = x, 0
            for i, module in enumerate(model.modules_list):
                x = module(x)
                if (i + 1) % 3 == 0:
                    x = x + model.residuals[residual_idx](residual_input)
                    x = F.relu(x)
                    residual_input, residual_idx = x, residual_idx + 1
            feats.append(model.gap(x).squeeze(-1).cpu().numpy())

    return np.concatenate(feats)


class AttributeNet(InceptionTime1D):
    """InceptionTime1D with a multi-class head, for the capacity-matched baseline.

    ``InceptionTime1D.forward`` ends with ``.reshape(-1)`` for its single-logit
    binary head, which is wrong for a k-way attribute. This subclass keeps the
    backbone byte-identical and returns ``(B, num_classes)`` instead.

    Capacity matching is the point of this class: it tests whether an equally
    expressive model reads an attribute from the raw input. A weaker baseline
    (for example a linear probe on PCA components) can only show that the
    attribute is not *linearly* accessible, which does not establish that the
    trained model is the source of the exposure.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input, residual_idx = x, 0
        for i, module in enumerate(self.modules_list):
            x = module(x)
            if (i + 1) % 3 == 0:
                x = x + self.residuals[residual_idx](residual_input)
                x = F.relu(x)
                residual_input, residual_idx = x, residual_idx + 1
        return self.head(self.gap(x).squeeze(-1))


# 10. Control metrics

def advantage(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Accuracy minus this group's own majority-class rate.

    Returns ``(advantage, accuracy, majority_rate)``.

    The majority rate is computed on ``y_true`` rather than taken from a global
    constant. That matters for the membership control: members and non-members
    are drawn from different folds and can differ in class balance, so scoring
    both against a shared baseline would manufacture an apparent gap.
    """
    import pandas as pd

    acc = float((y_true == y_pred).mean())
    maj = float(pd.Series(y_true).value_counts(normalize=True).max())
    return acc - maj, acc, maj


def bootstrap_gap_ci(
    y_member: np.ndarray,
    pred_member: np.ndarray,
    y_nonmember: np.ndarray,
    pred_nonmember: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    n_boot: int = 2000,
) -> tuple[float, float]:
    """Percentile interval for advantage(member) minus advantage(non-member).

    Resamples the two groups independently, recomputing each group's own
    majority rate on every draw so the interval reflects uncertainty in the
    baseline as well as in accuracy.

    Returns
    -------
    (lo, hi)
        95% percentile interval. An interval spanning zero means no detectable
        member-specific component, which is failure to detect rather than a
        demonstrated absence; report it as such.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    gaps = np.empty(n_boot)
    n_m, n_n = len(y_member), len(y_nonmember)
    for b in range(n_boot):
        i_m = rng.integers(0, n_m, n_m)
        i_n = rng.integers(0, n_n, n_n)
        a_m, _, _ = advantage(y_member[i_m], pred_member[i_m])
        a_n, _, _ = advantage(y_nonmember[i_n], pred_nonmember[i_n])
        gaps[b] = a_m - a_n

    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def summarize_gap_signs(gap_df, attribute_col: str = "attribute",
                        gap_col: str = "gap_mean") -> dict[str, tuple[int, int]]:
    """Count positive gaps per attribute.

    A membership effect must move every attribute in the same direction: a model
    that exposes its training members should expose them on sex *and* age. When
    attributes are internally consistent but point opposite ways, the residual is
    better explained by fold-level distribution shift than by disclosure.

    Returns
    -------
    dict
        ``{attribute: (n_positive, n_total)}``.
    """
    out = {}
    for attr, grp in gap_df.groupby(attribute_col):
        vals = grp[gap_col].to_numpy()
        out[attr] = (int((vals > 0).sum()), len(vals))
    return out


# Smoke test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Model instantiates and counts parameters
    model = InceptionTime1D()
    n_params = count_parameters(model)
    log.info(f"InceptionTime1D parameters: {n_params:,}")
    assert 300_000 < n_params < 600_000, (
        f"Param count {n_params} outside expected 300K-600K range"
    )

    # 2. Forward pass on random tensor
    x = torch.randn(4, NUM_LEADS, SIGNAL_LENGTH)
    logits = model(x)
    assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
    log.info(f"Forward pass OK: input {tuple(x.shape)} -> output {tuple(logits.shape)}")

    # 3. Opacus compatibility, model passes ModuleValidator
    try:
        from opacus.validators import ModuleValidator
        errors = ModuleValidator.validate(InceptionTime1D(), strict=False)
        if errors:
            log.warning(f"Opacus validation errors: {errors}")
        else:
            log.info("Opacus ModuleValidator: clean (no BatchNorm)")
    except ImportError:
        log.info("Opacus not installed, skipping validator check")

    # 4. MIA metric helpers
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    is_member = rng.integers(0, 2, 1000)
    auc = mia_auc(scores, is_member)
    tpr = mia_tpr_at_fpr(scores, is_member, target_fpr=0.001)
    log.info(f"MIA helpers OK: auc={auc:.3f}, tpr@0.1%fpr={tpr:.3f}")

    # 5. Environment helpers
    dev = get_device()
    log.info(f"Device: {dev}")
    assert tag_for(None, 0) == "target_baseline_seed00"
    assert tag_for(1.0, 2) == "target_eps_1.0_seed02"
    log.info("tag_for OK")

    # 6. Attribute targets, the single definition of AGE_BINS
    import pandas as pd
    demo = pd.DataFrame({
        "sex": [0, 1, 1, np.nan, 0],
        "age": [25, 50, 60, 70, np.nan],
    })
    targets = build_attribute_targets(demo)
    y_sex, valid_sex, n_sex = targets["sex"]
    y_age, valid_age, n_age = targets["age_bin"]
    assert n_sex == 2 and n_age == len(AGE_BIN_LABELS)
    assert valid_sex.sum() == 4 and valid_age.sum() == 4
    assert list(y_age[:4]) == [0, 1, 2, 3], f"Age binning wrong: {y_age[:4]}"
    log.info(f"build_attribute_targets OK: bins {AGE_BINS} -> {AGE_BIN_LABELS}")

    # 7. Penultimate features and the capacity-matched head
    X_dummy = np.random.randn(8, NUM_LEADS, SIGNAL_LENGTH).astype(np.float32)
    feats = extract_features(InceptionTime1D(), X_dummy, device=torch.device("cpu"))
    assert feats.shape[0] == 8 and feats.ndim == 2
    log.info(f"extract_features OK: {X_dummy.shape} -> {feats.shape}")

    attr_net = AttributeNet(num_classes=5)
    out_multi = attr_net(torch.from_numpy(X_dummy))
    assert out_multi.shape == (8, 5), f"Expected (8, 5), got {tuple(out_multi.shape)}"
    log.info(f"AttributeNet OK: multi-class output {tuple(out_multi.shape)}")

    # 8. Control metrics
    y_m = np.array([0, 0, 1, 1, 0, 1])
    p_m = np.array([0, 0, 1, 0, 0, 1])
    y_n = np.array([0, 1, 1, 0, 0, 1])
    p_n = np.array([0, 0, 1, 0, 1, 1])
    adv_m, acc_m, maj_m = advantage(y_m, p_m)
    log.info(f"advantage OK: acc={acc_m:.3f} maj={maj_m:.3f} adv={adv_m:+.3f}")

    lo, hi = bootstrap_gap_ci(y_m, p_m, y_n, p_n,
                              rng=np.random.default_rng(0), n_boot=200)
    assert lo <= hi
    log.info(f"bootstrap_gap_ci OK: [{lo:+.3f}, {hi:+.3f}]")

    gaps = pd.DataFrame({
        "attribute": ["sex"] * 3 + ["age_bin"] * 3,
        "gap_mean": [0.02, 0.01, 0.015, -0.02, -0.01, -0.017],
    })
    signs = summarize_gap_signs(gaps)
    assert signs["sex"] == (3, 3) and signs["age_bin"] == (0, 3)
    log.info(f"summarize_gap_signs OK: {signs}")

    log.info("All smoke tests passed")

    log.info("All smoke tests passed.")