from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import AuraNet, ProbModel


def export_onnx(model: AuraNet, path: Path, classes: list[str]) -> None:
    model = model.cpu().eval()
    wrapper = ProbModel(model).eval()
    dummy = torch.zeros(1, model.n_samples, model.n_channels, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            input_names=["emg"],
            output_names=["probs"],
            dynamic_axes={"emg": {0: "batch"}, "probs": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            input_names=["emg"],
            output_names=["probs"],
            dynamic_axes={"emg": {0: "batch"}, "probs": {0: "batch"}},
            opset_version=17,
        )
    _check_ort(path, dummy.numpy(), len(classes))


def _check_ort(path: Path, dummy, n_classes: int) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probs = sess.run(None, {"emg": dummy})[0]
    if probs.shape != (1, n_classes):
        raise RuntimeError(f"ONNX output shape {probs.shape}, expected (1, {n_classes})")
    total = float(probs.sum())
    if abs(total - 1.0) > 1e-3:
        raise RuntimeError(f"ONNX probabilities sum to {total}, expected 1")


def write_contract(path: Path, cfg: dict, mask: list[bool], split: dict[str, list[str]]) -> None:
    fs = float(cfg["sample_rate_hz"])
    win = int(round(float(cfg["window_sec"]) * fs))
    contract = {
        "format": "aura-hand-onnx-v1",
        "task": "6-class palm EMG, probabilities",
        "input_name": "emg",
        "input_layout": "NTC",
        "input_shape": [1, win, 8],
        "input_dtype": "float32",
        "output_name": "probs",
        "output_shape": [1, len(cfg["classes"])],
        "classes": list(cfg["classes"]),
        "sample_rate_hz": fs,
        "window_samples": win,
        "hop_samples": int(round(float(cfg["hop_sec"]) * fs)),
        "channel_names": [f"ch{i}" for i in range(1, 9)],
        "channel_keep": mask,
        "preprocess": {
            "in_graph": False,
            "unit": "microvolts, then per-channel robust scale",
            "scale": "median(abs(x)) / 0.6745, floor 1e-6, computed on the live calibration window",
            "filter": "causal sos: bandpass 20-120 order 4, bandstop 45-55 order 4, notch 100 Q=30",
            "note": "Apply the filter and scale before this model. Dropped channels may be left as-is; the graph zeros them.",
        },
        "imu_used": False,
        "optical_used_as_input": False,
        "split_unit": "session file, not window",
        "trained_on": split["train"],
        "early_stop_on": split["val"],
        "tested_on": split["test"],
        "test_used_for_weights": False,
    }
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
