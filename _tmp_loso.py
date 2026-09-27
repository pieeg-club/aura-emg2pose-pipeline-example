"""LOSO of the current classifier. Train+val pool, 32 epochs, test sealed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aura_pipeline.cli import _concat, _run_dir, _windows_by_session
from aura_pipeline.config import load_config
from aura_pipeline.data import discover, global_mask, load_session
from aura_pipeline.fit import balanced_accuracy, confusion, predict, set_seed, train_model

SHIP_EPOCHS = 32
SPLIT_PATH = Path("runs/20260927_092039/split.json")


def main() -> int:
    cfg = load_config()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    folder = Path("data/raw")
    print("load sessions", flush=True)
    sessions = []
    for path in discover(folder):
        print(f"  load {path.name}", flush=True)
        sessions.append(load_session(path, cfg))
    by_name = {s.name: s for s in sessions}
    mask = global_mask([by_name[n] for n in split["train"] if n in by_name], cfg)
    xs, ys, names = _windows_by_session(sessions, cfg, mask)
    pool = split["train"] + split["val"]
    out = _run_dir()
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(
        f"LOSO train+val only, {len(pool)} folds, epochs={SHIP_EPOCHS}. Test sealed.",
        flush=True,
    )
    scores = []
    for name in pool:
        i = names.index(name)
        others = [names.index(n) for n in pool if n != name]
        tr_x, tr_y = _concat([xs[j] for j in others], [ys[j] for j in others])
        print(f"  fold {name}  train windows {len(tr_y)}  hold {len(ys[i])}", flush=True)
        set_seed(int(cfg["seed"]))
        fold, info = train_model(tr_x, tr_y, None, None, cfg, mask, epochs=SHIP_EPOCHS)
        device = next(fold.parameters()).device
        pred = predict(fold, xs[i], int(cfg["batch_size"]), device)
        n_cls = len(cfg["classes"])
        score = balanced_accuracy(ys[i], pred, n_cls)
        mat = confusion(ys[i], pred, n_cls)
        row = {
            "session": name,
            "n": int(len(ys[i])),
            "balanced_accuracy": score,
            "confusion": mat.tolist(),
            "epochs": SHIP_EPOCHS,
            "device": info["device"],
        }
        scores.append(row)
        print(f"  {name} {score:.3f}", flush=True)
    acc = np.array([r["balanced_accuracy"] for r in scores], dtype=np.float64)
    summary = {
        "protocol": "current classifier, LOSO on train+val, test sealed",
        "command": "LOSO 32 epochs, no early stop, seed 7, same AuraNet",
        "ship_epochs": SHIP_EPOCHS,
        "split": split,
        "pool": pool,
        "scores": scores,
        "mean": float(acc.mean()),
        "std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "min": float(acc.min()),
        "max": float(acc.max()),
        "channel_keep": mask,
    }
    (out / "loso.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# LOSO current classifier",
        "",
        "Leave-one-session-out on train+val only. Test sessions stay sealed.",
        f"Epochs per fold: {SHIP_EPOCHS}. No early stop. Seed 7.",
        "",
        f"- Mean bal-acc: {summary['mean'] * 100:.1f}%",
        f"- Std: {summary['std'] * 100:.1f} pp",
        f"- Min: {summary['min'] * 100:.1f}% ({scores[int(acc.argmin())]['session']})",
        f"- Max: {summary['max'] * 100:.1f}% ({scores[int(acc.argmax())]['session']})",
        "",
        "| session | n | bal-acc |",
        "| --- | --- | --- |",
    ]
    for row in scores:
        lines.append(f"| {row['session']} | {row['n']} | {row['balanced_accuracy'] * 100:.1f}% |")
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'loso.json'}", flush=True)
    print(
        f"mean {summary['mean']:.3f}  std {summary['std']:.3f}  "
        f"min {summary['min']:.3f}  max {summary['max']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
