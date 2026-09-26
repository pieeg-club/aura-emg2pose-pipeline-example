from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_qc(path: Path, rows: list[dict], mask: list[bool]) -> None:
    lines = [
        "# Channel QC",
        "",
        "Hard failures (flat, empty, railed, or forced) are zeroed in that file.",
        "A channel is removed from the shipped model only if it fails in many files.",
        "",
        "Shipped channel keep mask (ch1..ch8): " + " ".join("1" if k else "0" for k in mask),
        "",
        "| session | ch | std uV | snr | mains | verdict | reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        snr = "" if row["snr"] is None else f"{row['snr']:.2f}"
        mains = "" if row["mains_ratio"] is None else f"{row['mains_ratio']:.2f}"
        lines.append(
            f"| {row['session']} | {row['channel']} | {row['std_uv']:.1f} | {snr} | {mains} | {row['verdict']} | {row['reason']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary: dict) -> None:
    counts = summary["class_counts"]
    lines = [
        "# Aura hand training run",
        "",
        f"- Train sessions ({len(summary['split']['train'])}): {', '.join(summary['split']['train'])}",
        f"- Val sessions ({len(summary['split']['val'])}): {', '.join(summary['split']['val'])}",
        f"- Test sessions ({len(summary['split']['test'])}): {', '.join(summary['split']['test'])}",
        f"- Train windows: {summary['n_train']}",
        f"- Val balanced accuracy (early stop only): {_pct(summary.get('val_balanced_accuracy'))}",
        f"- Test balanced accuracy (never used to train): {_pct(summary.get('test_balanced_accuracy'))}",
        f"- Epoch kept: {summary['ship_epochs']}",
        f"- Device: {summary['device']}",
        f"- ONNX: `{summary['onnx']}`",
        "",
        "## Train class counts",
        "",
        "| class | windows |",
        "| --- | --- |",
    ]
    for name, n in counts.items():
        lines.append(f"| {name} | {n} |")
    lines += [
        "",
        "## Test confusion",
        "",
        "Rows are true class, columns are predicted, order is the class list in config.yaml.",
        "This matrix is the test sessions only. Those files were not in the loss and not in early stopping.",
        "",
        "```",
        np.array2string(np.asarray(summary["confusion"])) if summary.get("confusion") is not None else "n/a",
        "```",
        "",
        "Copy `split.json` into config.yaml (`train_sessions`, `val_sessions`, `test_sessions`) to freeze this split.",
        "",
    ]
    if summary.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for w in summary["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"
