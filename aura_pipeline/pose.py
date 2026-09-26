"""Continuous hand-pose regression from palm EMG.

Parallel to the classifier. The target is the optical hand tracker carried in each
session file (per-finger curls plus pinch and openness), not motion capture. The model
distils that optical stream into the 8-channel EMG, so its ceiling is the tracker's own
quality. Output is one pose vector per window, each DoF in [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import N_CH, apply_mask, pose_target_names
from .fit import set_seed
from .model import AuraNet


class PoseNet(nn.Module):
    """AuraNet backbone with a sigmoid head. Output is a pose vector in [0, 1]."""

    def __init__(
        self,
        n_dof: int,
        n_channels: int = N_CH,
        n_samples: int = 250,
        channel_mask: list[bool] | None = None,
    ):
        super().__init__()
        self.net = AuraNet(
            n_classes=n_dof,
            n_channels=n_channels,
            n_samples=n_samples,
            channel_mask=channel_mask,
        )
        self.n_dof = n_dof
        self.n_channels = n_channels
        self.n_samples = n_samples

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(emg))


def make_pose_windows(
    session, cfg: dict, names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows of EMG paired with a confidence-weighted pose target.

    A window is kept when at least half its samples carry a valid, present pose and the
    median optical confidence clears ``pose_min_conf``. The target is the per-DoF median
    over the valid samples; the sample weight is that median confidence.
    """
    n_dof = len(names)
    win = int(round(float(cfg["window_sec"]) * session.fs))
    empty = (
        np.empty((0, win, N_CH), np.float32),
        np.empty((0, n_dof), np.float32),
        np.empty((0,), np.float32),
    )
    if session.pose is None or session.pose_conf is None:
        return empty
    hop = int(round(float(cfg["hop_sec"]) * session.fs))
    min_conf = float(cfg.get("pose_min_conf", 0.3))
    pose = session.pose
    conf = session.pose_conf
    emg = session.emg
    n = min(len(emg), len(pose), len(conf))
    xs, ys, ws = [], [], []
    for start in range(0, n - win + 1, hop):
        seg = pose[start:start + win]
        c = conf[start:start + win]
        valid = np.isfinite(seg).all(axis=1) & np.isfinite(c)
        if valid.mean() < 0.5:
            continue
        cv = c[valid]
        med_conf = float(np.median(cv))
        if med_conf < min_conf:
            continue
        xs.append(emg[start:start + win])
        ys.append(np.median(seg[valid], axis=0).astype(np.float32))
        ws.append(med_conf)
    if not xs:
        return empty
    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.asarray(ws, dtype=np.float32),
    )


def pose_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    per = []
    for j, name in enumerate(names):
        t = y_true[:, j]
        p = y_pred[:, j]
        mae = float(np.mean(np.abs(t - p))) if len(t) else float("nan")
        if len(t) > 1 and np.std(t) > 1e-6 and np.std(p) > 1e-6:
            r = float(np.corrcoef(t, p)[0, 1])
        else:
            r = float("nan")
        per.append({"dof": name, "mae": mae, "r": r})
    maes = [d["mae"] for d in per if np.isfinite(d["mae"])]
    rs = [d["r"] for d in per if np.isfinite(d["r"])]
    return {
        "per_dof": per,
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "r": float(np.mean(rs)) if rs else float("nan"),
    }


@torch.no_grad()
def predict_pose(model: nn.Module, x: np.ndarray, batch: int, device: torch.device) -> np.ndarray:
    model.eval()
    n_dof = getattr(model, "n_dof", 0)
    if len(x) == 0:
        return np.empty((0, n_dof), np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch, shuffle=False)
    out = [model(xb.to(device)).cpu().numpy() for (xb,) in loader]
    return np.concatenate(out)


def train_pose(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    cfg: dict,
    names: list[str],
    channel_mask: list[bool],
    epochs: int | None = None,
) -> tuple[PoseNet, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseNet(len(names), x_tr.shape[2], x_tr.shape[1], channel_mask).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loss_fn = nn.SmoothL1Loss(beta=float(cfg.get("pose_loss_beta", 0.1)), reduction="none")
    ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr), torch.from_numpy(w_tr))
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    max_epochs = int(epochs if epochs is not None else cfg["epochs"])
    patience = int(cfg["patience"])
    best_state = None
    best_mae = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            per = loss_fn(model(xb), yb).mean(dim=1)
            loss = (per * wb).sum() / (wb.sum() + 1e-8)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            seen += len(xb)
        row = {"epoch": epoch, "loss": total / max(1, seen)}
        if x_val is not None and len(x_val):
            metrics = pose_metrics(y_val, predict_pose(model, x_val, int(cfg["batch_size"]), device), names)
            row["val_mae"] = metrics["mae"]
            row["val_r"] = metrics["r"]
            if metrics["mae"] < best_mae - 1e-5:
                best_mae = metrics["mae"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        else:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        vmae = f"{row['val_mae']:.4f}" if "val_mae" in row else "-"
        vr = f"{row['val_r']:.3f}" if "val_r" in row else "-"
        print(f"  epoch {epoch:02d}  loss {row['loss']:.4f}  val mae {vmae}  val r {vr}", flush=True)
        if x_val is not None and stale >= patience:
            print(f"  early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    info = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_mae": None if best_mae == float("inf") else best_mae,
        "history": history,
    }
    return model, info


def export_pose_onnx(model: PoseNet, path: Path, names: list[str]) -> None:
    model = model.cpu().eval()
    dummy = torch.zeros(1, model.n_samples, model.n_channels, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["emg"],
        output_names=["pose"],
        dynamic_axes={"emg": {0: "batch"}, "pose": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)
    _check_ort(path, dummy.numpy(), len(names))


def _check_ort(path: Path, dummy, n_dof: int) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"emg": dummy})[0]
    if out.shape != (1, n_dof):
        raise RuntimeError(f"ONNX pose shape {out.shape}, expected (1, {n_dof})")
    if float(out.min()) < -1e-4 or float(out.max()) > 1.0 + 1e-4:
        raise RuntimeError("ONNX pose outputs fall outside [0, 1]")


def write_pose_contract(path: Path, cfg: dict, names: list[str], mask: list[bool], split: dict) -> None:
    fs = float(cfg["sample_rate_hz"])
    win = int(round(float(cfg["window_sec"]) * fs))
    contract = {
        "format": "aura-pose-onnx-v1",
        "task": "continuous hand-pose regression from palm EMG",
        "input_name": "emg",
        "input_layout": "NTC",
        "input_shape": [1, win, N_CH],
        "input_dtype": "float32",
        "output_name": "pose",
        "output_shape": [1, len(names)],
        "output_range": [0, 1],
        "dof": list(names),
        "sample_rate_hz": fs,
        "window_samples": win,
        "hop_samples": int(round(float(cfg["hop_sec"]) * fs)),
        "channel_names": [f"ch{i}" for i in range(1, N_CH + 1)],
        "channel_keep": mask,
        "preprocess": {
            "in_graph": False,
            "unit": "microvolts, then per-channel robust scale",
            "scale": "median(abs(x)) / 0.6745, floor 1e-6, per recording over the filtered signal",
            "filter": "causal sos: bandpass 20-120 order 4, bandstop 45-55 order 4, notch 100 Q=30",
            "note": "Apply the filter and scale before this model. Dropped channels may be left as-is; the graph zeros them.",
        },
        "target_source": "optical hand tracker carried in the session file, not motion capture",
        "imu_used": False,
        "split_unit": "session file, not window",
        "trained_on": split["train"],
        "early_stop_on": split["val"],
        "tested_on": split["test"],
        "test_used_for_weights": False,
    }
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _fmt(value) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.4f}"


def write_pose_report(
    path: Path,
    names: list[str],
    split: dict,
    val_metrics: dict | None,
    test_metrics: dict | None,
    info: dict,
    mask: list[bool],
    counts: dict,
) -> None:
    lines = [
        "# Aura hand-pose regression run",
        "",
        "Target is the optical hand tracker carried in the session files, not motion capture.",
        "The model distils that stream into EMG, so its ceiling is the tracker's quality.",
        "",
        f"- Train sessions ({len(split['train'])}): {', '.join(split['train'])}",
        f"- Val sessions ({len(split['val'])}): {', '.join(split['val'])}",
        f"- Test sessions ({len(split['test'])}): {', '.join(split['test'])}",
        f"- Windows: train={counts['train']} val={counts['val']} test={counts['test']}",
        f"- Val mean MAE (early stop only): {_fmt(val_metrics['mae']) if val_metrics else 'n/a'}",
        f"- Test mean MAE (never used to train): {_fmt(test_metrics['mae']) if test_metrics else 'n/a'}",
        f"- Test mean r: {_fmt(test_metrics['r']) if test_metrics else 'n/a'}",
        f"- Epoch kept: {info['best_epoch']}",
        f"- Device: {info['device']}",
        "",
        "## Per-DoF test error",
        "",
        "| DoF | MAE | r |",
        "| --- | --- | --- |",
    ]
    test_per = {d["dof"]: d for d in (test_metrics["per_dof"] if test_metrics else [])}
    for name in names:
        row = test_per.get(name)
        mae = _fmt(row["mae"]) if row else "n/a"
        r = _fmt(row["r"]) if row else "n/a"
        lines.append(f"| {name} | {mae} | {r} |")
    lines += [
        "",
        "MAE is in target units (each DoF is [0, 1]). r is Pearson correlation on the test sessions.",
        "Copy `split.json` into config.yaml to freeze this split.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pose(sessions, cfg: dict, out: Path, split: dict, mask: list[bool]) -> int:
    names = pose_target_names(cfg)
    if not any(s.pose is not None for s in sessions):
        raise SystemExit("No pose columns found. Expected: " + ", ".join(names))

    per_session = {}
    for session in sessions:
        x, y, w = make_pose_windows(session, cfg, names)
        x = apply_mask(x, mask)
        per_session[session.name] = (x, y, w)
        print(f"  {session.name}: {len(x)} pose windows", flush=True)

    win = int(round(float(cfg["window_sec"]) * float(cfg["sample_rate_hz"])))

    def gather(role: str):
        xs, ys, ws = [], [], []
        for name in split[role]:
            bundle = per_session.get(name)
            if bundle is not None and len(bundle[0]):
                xs.append(bundle[0])
                ys.append(bundle[1])
                ws.append(bundle[2])
        if not xs:
            return (
                np.empty((0, win, N_CH), np.float32),
                np.empty((0, len(names)), np.float32),
                np.empty((0,), np.float32),
            )
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)

    x_tr, y_tr, w_tr = gather("train")
    x_val, y_val, w_val = gather("val")
    x_te, y_te, w_te = gather("test")
    if not len(x_tr):
        raise SystemExit("No pose training windows. Lower pose_min_conf or check the optical columns.")
    counts = {"train": int(len(x_tr)), "val": int(len(x_val)), "test": int(len(x_te))}
    print(f"pose windows  train={counts['train']}  val={counts['val']}  test={counts['test']}", flush=True)

    set_seed(int(cfg["seed"]))
    model, info = train_pose(x_tr, y_tr, w_tr, x_val, y_val, cfg, names, mask)
    device = next(model.parameters()).device
    val_metrics = (
        pose_metrics(y_val, predict_pose(model, x_val, int(cfg["batch_size"]), device), names)
        if len(x_val)
        else None
    )
    test_metrics = (
        pose_metrics(y_te, predict_pose(model, x_te, int(cfg["batch_size"]), device), names)
        if len(x_te)
        else None
    )
    if val_metrics:
        print(f"val  mean mae {val_metrics['mae']:.4f}  mean r {val_metrics['r']:.3f}  (early stop only)", flush=True)
    if test_metrics:
        print(f"test mean mae {test_metrics['mae']:.4f}  mean r {test_metrics['r']:.3f}  (not used for weights)", flush=True)

    onnx_path = out / "pose.onnx"
    export_pose_onnx(model, onnx_path, names)
    torch.save(
        {"state_dict": model.cpu().state_dict(), "dof": names, "mask": mask, "split": split},
        out / "pose.pt",
    )
    write_pose_contract(out / "pose_contract.json", cfg, names, mask, split)
    write_pose_report(out / "pose_report.md", names, split, val_metrics, test_metrics, info, mask, counts)
    summary = {
        "dof": names,
        "counts": counts,
        "split": split,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": info["best_epoch"],
        "device": info["device"],
        "channel_keep": mask,
        "onnx": str(onnx_path),
    }
    (out / "pose_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {onnx_path}")
    print(f"contract {out / 'pose_contract.json'}")
    return 0
