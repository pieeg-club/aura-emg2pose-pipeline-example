from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import ROOT, load_config, resolve_dir
from .data import apply_mask, discover, global_mask, load_session, make_windows
from .export import export_onnx, write_contract
from .fit import balanced_accuracy, confusion, predict, set_seed, train_model
from .report import write_qc, write_report
from .split import assign_sessions

warnings.filterwarnings("ignore", message="Workbook contains no default style")


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Aura hand model and write ONNX.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--data", type=Path, default=None, help="Folder of xlsx/csv sessions.")
    p.add_argument("--qc-only", action="store_true", help="Channel report only. No training.")
    p.add_argument("--self-test", action="store_true", help="Train on synthetic EMG and export ONNX.")
    p.add_argument("--loso", action="store_true", help="Also score leave-one-session-out. Slow.")
    p.add_argument("--pose", action="store_true", help="Train continuous hand-pose regression, not the classifier.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--test", action="append", default=None, help="Pin a session into the test split. Repeatable.")
    return p.parse_args(argv)


def _data_dir(cfg: dict, override: Path | None) -> Path:
    if override is not None:
        path = override if override.is_absolute() else (Path.cwd() / override)
        path = path.resolve()
        if not discover(path):
            raise SystemExit(f"No xlsx or csv in {path}")
        return path
    primary = resolve_dir(ROOT, cfg["data_dir"])
    if primary and discover(primary):
        return primary
    fallback = resolve_dir(ROOT, cfg.get("fallback_data_dir"))
    if fallback and discover(fallback):
        print(f"data/raw is empty. Using {fallback}")
        return fallback
    raise SystemExit(
        "No recordings found. Put session xlsx or csv files in data/raw, or pass --data <folder>."
    )


def _run_dir() -> Path:
    path = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "latest.txt").write_text(str(path), encoding="utf-8")
    return path


def _bundle(cfg: dict, folder: Path):
    files = discover(folder)
    print(f"{len(files)} session file(s) in {folder}", flush=True)
    sessions = []
    for path in files:
        print(f"  load {path.name}", flush=True)
        sessions.append(load_session(path, cfg))
    return sessions


def _windows_by_session(sessions, cfg, mask):
    xs, ys, names = [], [], []
    for session in sessions:
        x, y = make_windows(session, cfg)
        x = apply_mask(x, mask)
        if len(x) == 0:
            print(f"  no clean windows in {session.name}", flush=True)
            continue
        print(f"  {session.name}: {len(x)} windows", flush=True)
        xs.append(x)
        ys.append(y)
        names.append(session.name)
    return xs, ys, names


def _concat(xs, ys):
    if not xs:
        win = 250
        return np.empty((0, win, 8), np.float32), np.empty((0,), np.int64)
    return np.concatenate(xs), np.concatenate(ys)


def _check_classes(y: np.ndarray, classes: list[str]) -> None:
    missing = [name for i, name in enumerate(classes) if np.sum(y == i) == 0]
    if missing:
        raise SystemExit(
            "No windows for: " + ", ".join(missing) + ". Check guided labels before training."
        )


def run_self_test() -> int:
    from .model import AuraNet

    cfg = load_config()
    cfg["epochs"] = 1
    cfg["patience"] = 1
    cfg["batch_size"] = 16
    fs = 250
    win = 250
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for cls in range(len(cfg["classes"])):
        block = rng.normal(0, 0.2, size=(24, win, 8)).astype(np.float32)
        block[:, :, cls % 8] += 1.5 + 0.3 * cls
        xs.append(block)
        ys.append(np.full(len(block), cls, np.int64))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mask = [True] * 8
    mask[7] = False
    x[:, :, 7] = 0.0
    set_seed(int(cfg["seed"]))
    model, info = train_model(x, y, x[:32], y[:32], cfg, mask, epochs=1)
    out = _run_dir()
    onnx_path = out / "model.onnx"
    export_onnx(model, onnx_path, cfg["classes"])
    write_contract(out / "contract.json", cfg, mask, {"train": ["synthetic"], "val": [], "test": []})
    probe = AuraNet(channel_mask=mask)
    probe.load_state_dict(model.cpu().state_dict())
    print(f"self-test ok  device={info['device']}  onnx={onnx_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.self_test:
        return run_self_test()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.loso:
        cfg["loso"] = True
    if args.test:
        cfg["test_sessions"] = list(args.test)

    folder = _data_dir(cfg, args.data)
    sessions = _bundle(cfg, folder)
    mask = global_mask(sessions, cfg)
    out = _run_dir()
    rows = [row for s in sessions for row in s.qc_rows]
    write_qc(out / "qc.md", rows, mask)
    (out / "qc.json").write_text(json.dumps({"mask": mask, "rows": rows}, indent=2), encoding="utf-8")
    print("channel keep: " + " ".join(f"ch{i+1}" for i, k in enumerate(mask) if k), flush=True)
    print(f"qc report: {out / 'qc.md'}", flush=True)
    if args.qc_only:
        return 0

    by_name = {s.name: s for s in sessions if s.name}
    split = assign_sessions([s.name for s in sessions], cfg)
    # Channel mask is fit on train only. Validation and test must not decide which channels exist.
    mask = global_mask([by_name[n] for n in split["train"] if n in by_name], cfg)
    write_qc(out / "qc.md", rows, mask)
    (out / "qc.json").write_text(json.dumps({"mask": mask, "rows": rows, "split": split}, indent=2), encoding="utf-8")
    print("channel keep: " + " ".join(f"ch{i+1}" for i, k in enumerate(mask) if k), flush=True)
    print(
        "split  train={train}  val={val}  test={test}".format(**{k: ",".join(v) for k, v in split.items()}),
        flush=True,
    )
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    if args.pose:
        from .pose import run_pose

        return run_pose(sessions, cfg, out, split, mask)

    xs, ys, names = _windows_by_session(sessions, cfg, mask)
    if not names:
        raise SystemExit("No trainable windows. See the QC report.")
    have = set(names)
    for role, group in split.items():
        missing = [n for n in group if n not in have]
        if missing:
            raise SystemExit(f"{role} sessions have no clean windows: {', '.join(missing)}")
    groups = {}
    for role in ("train", "val", "test"):
        chosen = [names.index(n) for n in split[role]]
        groups[role] = _concat([xs[i] for i in chosen], [ys[i] for i in chosen])
    x_tr, y_tr = groups["train"]
    x_val, y_val = groups["val"]
    x_te, y_te = groups["test"]
    _check_classes(y_tr, cfg["classes"])
    counts = {name: int(np.sum(y_tr == i)) for i, name in enumerate(cfg["classes"])}
    print("train windows " + ", ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

    set_seed(int(cfg["seed"]))
    model, info = train_model(x_tr, y_tr, x_val, y_val, cfg, mask)
    device = next(model.parameters()).device
    val_pred = predict(model, x_val, int(cfg["batch_size"]), device)
    val_score = balanced_accuracy(y_val, val_pred, len(cfg["classes"]))
    test_pred = predict(model, x_te, int(cfg["batch_size"]), device)
    test_score = balanced_accuracy(y_te, test_pred, len(cfg["classes"]))
    conf = confusion(y_te, test_pred, len(cfg["classes"]))
    print(f"val balanced accuracy {val_score:.3f}  (early stop only)", flush=True)
    print(f"test balanced accuracy {test_score:.3f}  (not used for weights)", flush=True)
    ship_epochs = max(1, int(info["best_epoch"]))

    if cfg.get("loso"):
        pool = split["train"] + split["val"]
        print("leave-one-session-out on train+val only. Test stays sealed.", flush=True)
        scores = []
        for name in pool:
            i = names.index(name)
            others = [names.index(n) for n in pool if n != name]
            tr_x, tr_y = _concat([xs[j] for j in others], [ys[j] for j in others])
            # The held-out session is scored only. It is not used for early stopping.
            fold, _ = train_model(tr_x, tr_y, None, None, cfg, mask, epochs=ship_epochs)
            pred = predict(fold, xs[i], int(cfg["batch_size"]), next(fold.parameters()).device)
            score = balanced_accuracy(ys[i], pred, len(cfg["classes"]))
            scores.append({"session": name, "balanced_accuracy": score})
            print(f"  {name} {score:.3f}", flush=True)
        (out / "loso.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    onnx_path = out / "model.onnx"
    export_onnx(model, onnx_path, cfg["classes"])
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "classes": cfg["classes"],
            "mask": mask,
            "split": split,
        },
        out / "model.pt",
    )
    write_contract(out / "contract.json", cfg, mask, split)
    summary = {
        "n_sessions": len(names),
        "n_windows": int(len(y_tr) + len(y_val) + len(y_te)),
        "n_train": int(len(y_tr)),
        "class_counts": counts,
        "split": split,
        "val_balanced_accuracy": val_score,
        "test_balanced_accuracy": test_score,
        "confusion": conf.tolist(),
        "ship_epochs": ship_epochs,
        "device": info["device"],
        "onnx": str(onnx_path),
        "channel_keep": mask,
        "warnings": [w for s in sessions for w in s.warnings],
    }
    write_report(out / "report.md", summary)
    print(f"wrote {onnx_path}")
    print(f"contract {out / 'contract.json'}")
    return 0
