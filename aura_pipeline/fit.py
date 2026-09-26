from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import AuraNet


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        scores.append(float(np.mean(y_pred[mask] == c)))
    return float(np.mean(scores)) if scores else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat


def _loader(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    loader = _loader(x, np.zeros(len(x), np.int64), batch, shuffle=False)
    for xb, _ in loader:
        logits = model(xb.to(device))
        out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,), np.int64)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    cfg: dict,
    channel_mask: list[bool],
    epochs: int | None = None,
) -> tuple[AuraNet, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(cfg["classes"])
    win = x_train.shape[1]
    model = AuraNet(
        n_classes=n_classes,
        n_channels=x_train.shape[2],
        n_samples=win,
        channel_mask=channel_mask,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, n_classes).to(device))
    loader = _loader(x_train, y_train, int(cfg["batch_size"]), shuffle=True)
    max_epochs = int(epochs if epochs is not None else cfg["epochs"])
    patience = int(cfg["patience"])
    best_state = None
    best_score = -1.0
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            seen += len(xb)
        row = {"epoch": epoch, "loss": total / max(1, seen)}
        if x_val is not None and len(x_val):
            pred = predict(model, x_val, int(cfg["batch_size"]), device)
            score = balanced_accuracy(y_val, pred, n_classes)
            row["val_balanced_accuracy"] = score
            if score > best_score + 1e-4:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        else:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        score_txt = f"{row['val_balanced_accuracy']:.3f}" if "val_balanced_accuracy" in row else "-"
        print(f"  epoch {epoch:02d}  loss {row['loss']:.4f}  val bal-acc {score_txt}", flush=True)
        if x_val is not None and stale >= patience:
            print(f"  early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    info = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_balanced_accuracy": None if best_score < 0 else best_score,
        "history": history,
    }
    return model, info
