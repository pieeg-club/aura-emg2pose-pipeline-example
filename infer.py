"""Mock one live window. Filter and scale are the caller's job; they are not in the ONNX graph."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from aura_pipeline.config import ROOT, load_config
from aura_pipeline.signal import causal_filter, design_sos, robust_scale

WIN = 250
N_CH = 8
WARMUP = 125  # 0.5 s at 250 Hz. Training drops this after the causal filter.


def resolve_onnx(arg: str | None) -> Path:
    if arg:
        path = Path(arg)
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    latest = ROOT / "runs" / "latest.txt"
    if latest.is_file():
        cand = Path(latest.read_text(encoding="utf-8").strip()) / "model.onnx"
        if cand.is_file():
            return cand
    found = sorted((ROOT / "runs").glob("*/model.onnx"))
    if found:
        return found[-1]
    raise SystemExit("No model.onnx. Train with python train.py, or pass a path.")


def main() -> int:
    cfg = load_config()
    onnx_path = resolve_onnx(sys.argv[1] if len(sys.argv) > 1 else None)
    if not onnx_path.is_file():
        raise SystemExit(f"Missing {onnx_path}")
    contract_path = onnx_path.parent / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else {}
    classes = list(contract.get("classes") or cfg["classes"])
    input_name = str(contract.get("input_name") or "emg")
    win = int(contract.get("window_samples") or WIN)

    # Mock live EMG in microvolts, NTC (time, channel). Extra warmup matches training.
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 25, size=(WARMUP + win, N_CH)).astype(np.float32)
    filtered = causal_filter(raw, design_sos(float(cfg["sample_rate_hz"]), cfg))[WARMUP:]
    window = (filtered / robust_scale(filtered)).astype(np.float32)[None, :, :]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probs = sess.run(None, {input_name: window})[0][0]
    print(f"onnx {onnx_path}")
    print(f"input {tuple(window.shape)} float32  time x channel  scaled EMG")
    for name, p in zip(classes, probs):
        print(f"  {name:10s} {float(p):.4f}")
    print(f"sum {float(probs.sum()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
