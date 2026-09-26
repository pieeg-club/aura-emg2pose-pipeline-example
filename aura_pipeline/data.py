from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .signal import causal_filter, design_sos, mains_ratio, robust_scale, to_rate

N_CH = 8
CHANNEL_NAMES = [f"ch{i}" for i in range(1, N_CH + 1)]
LABEL_ALIASES = {
    "open-hand": "open",
    "openhand": "open",
    "thumbup": "thumb-up",
    "thumbs-up": "thumb-up",
    "thumb-up": "thumb-up",
}


@dataclass
class Session:
    name: str
    path: str
    fs: float
    emg_uv: np.ndarray
    emg: np.ndarray
    labels: np.ndarray
    scale: np.ndarray
    hard_fail: list[bool]
    qc_rows: list[dict] = field(default_factory=list)
    n_raw: int = 0
    n_used: int = 0
    warnings: list[str] = field(default_factory=list)


def discover(folder: Path) -> list[Path]:
    files = []
    for path in sorted(folder.iterdir() if folder.is_dir() else []):
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            files.append(path)
    return files


def _norm_label(value, classes: set[str]) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    text = LABEL_ALIASES.get(text, text)
    return text if text in classes else None


def _flag(series: pd.Series | None, n: int) -> np.ndarray:
    if series is None:
        return np.zeros(n, dtype=bool)
    num = pd.to_numeric(series, errors="coerce").to_numpy()
    return np.nan_to_num(num, nan=0.0) > 0.5


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=0)


def _hard_fail(ch: np.ndarray) -> tuple[bool, str]:
    finite = ch[np.isfinite(ch)]
    if finite.size < 16:
        return True, "empty"
    std = float(np.std(finite))
    ptp = float(np.ptp(finite))
    if std < 1.0 or ptp < 2.0:
        return True, "flat"
    # Stuck near one rail.
    close = np.mean((np.abs(finite - finite.min()) < 1.0) | (np.abs(finite - finite.max()) < 1.0))
    if ptp > 50.0 and close > 0.98:
        return True, "railed"
    return False, "ok"


def load_session(path: Path, cfg: dict) -> Session:
    classes = set(cfg["classes"])
    df = load_table(path)
    missing = [c for c in ["t", *CHANNEL_NAMES, "label"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")

    if "emg_new" in df.columns:
        keep = pd.to_numeric(df["emg_new"], errors="coerce").fillna(0).to_numpy() > 0.5
        df = df.loc[keep].reset_index(drop=True)

    raw = df[CHANNEL_NAMES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite_row = np.isfinite(raw).all(axis=1)
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=np.float64)
    finite_row &= np.isfinite(t)
    df = df.loc[finite_row].reset_index(drop=True)
    raw = raw[finite_row]
    t = t[finite_row]
    n_raw = int(len(df))

    labels = np.array([_norm_label(v, classes) for v in df["label"]], dtype=object)
    if "label_source" in df.columns:
        source = df["label_source"].astype(str).str.strip().str.lower().to_numpy()
        guided = source == "guided"
        labels = np.where(guided, labels, None)

    if cfg.get("drop_optical_disagreement", True) and "optical_class" in df.columns:
        present = _flag(df["optical_present"] if "optical_present" in df.columns else None, len(df))
        conf = pd.to_numeric(df["optical_conf"], errors="coerce").fillna(0).to_numpy() if "optical_conf" in df.columns else np.ones(len(df))
        optical = np.array([_norm_label(v, classes) for v in df["optical_class"]], dtype=object)
        disagree = present & (conf >= float(cfg.get("optical_conf_min", 0.6)))
        for i in range(len(labels)):
            if not disagree[i] or labels[i] is None or optical[i] is None:
                continue
            pair = {labels[i], optical[i]}
            if pair == {"rest", "open"}:
                continue
            if optical[i] != labels[i]:
                labels[i] = None

    target_fs = float(cfg["sample_rate_hz"])
    raw32, fs, resampled_n = to_rate(raw.astype(np.float32), t, target_fs)
    warnings = []
    if resampled_n is not None:
        idx = np.linspace(0, len(labels) - 1, resampled_n)
        labels = labels[np.clip(np.round(idx).astype(int), 0, len(labels) - 1)]
        warnings.append(f"resampled toward {target_fs:.0f} Hz")
    elif abs(fs - target_fs) / target_fs > 0.005:
        warnings.append(f"sample rate {fs:.1f} Hz, filtered as {target_fs:.0f}")

    mains = mains_ratio(raw32, target_fs, float(cfg.get("notch_mains_hz", 50.0)))
    filtered = causal_filter(raw32, design_sos(target_fs, cfg))
    warmup = int(0.5 * target_fs)
    if len(filtered) > warmup + 8:
        filtered = filtered[warmup:]
        labels = labels[warmup:]
        raw32 = raw32[warmup:]

    forced = {int(c) for c in cfg.get("drop_channels") or []}
    hard = []
    rows = []
    rest = labels == "rest"
    active = np.array([v is not None and v != "rest" for v in labels])
    for i in range(N_CH):
        failed, reason = _hard_fail(filtered[:, i])
        if (i + 1) in forced:
            failed, reason = True, "forced"
        snr = np.nan
        if rest.sum() > 8 and active.sum() > 8:
            snr = float(np.sqrt(np.mean(filtered[active, i] ** 2)) / (np.sqrt(np.mean(filtered[rest, i] ** 2)) + 1e-6))
        hard.append(failed)
        rows.append({
            "session": path.stem,
            "channel": i + 1,
            "std_uv": float(np.std(filtered[:, i])),
            "snr": None if not np.isfinite(snr) else round(snr, 3),
            "mains_ratio": None if not np.isfinite(mains[i]) else round(float(mains[i]), 3),
            "verdict": "drop" if failed else "keep",
            "reason": reason,
        })

    masked = filtered.copy()
    for i, failed in enumerate(hard):
        if failed:
            masked[:, i] = 0.0
    scale = robust_scale(masked)
    scaled = (masked / scale).astype(np.float32)

    return Session(
        name=path.stem,
        path=str(path),
        fs=target_fs,
        emg_uv=filtered.astype(np.float32),
        emg=scaled,
        labels=labels,
        scale=scale,
        hard_fail=hard,
        qc_rows=rows,
        n_raw=n_raw,
        n_used=int(len(scaled)),
        warnings=warnings,
    )


def trim_transitions(labels: np.ndarray, edge: int) -> np.ndarray:
    out = labels.copy()
    if edge <= 0 or len(out) < 2:
        return out
    change = np.zeros(len(out), dtype=bool)
    change[1:] = out[1:] != out[:-1]
    for i in np.flatnonzero(change):
        lo = max(0, i - edge)
        hi = min(len(out), i + edge)
        out[lo:hi] = None
    return out


def make_windows(session: Session, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    fs = session.fs
    win = int(round(float(cfg["window_sec"]) * fs))
    hop = int(round(float(cfg["hop_sec"]) * fs))
    edge = int(round(float(cfg["edge_trim_sec"]) * fs))
    if win < 128 or hop < 1:
        raise ValueError("window_sec must cover at least 128 samples and hop_sec must be positive")
    labels = trim_transitions(session.labels, edge)
    xs, ys = [], []
    class_to_i = {name: i for i, name in enumerate(cfg["classes"])}
    min_frac = float(cfg.get("min_label_fraction", 0.9))
    for start in range(0, len(labels) - win + 1, hop):
        chunk = labels[start:start + win]
        named = [v for v in chunk if v is not None]
        if len(named) / win < min_frac:
            continue
        name = max(set(named), key=named.count)
        if named.count(name) / win < min_frac:
            continue
        if any(v not in (None, name) for v in chunk):
            continue
        xs.append(session.emg[start:start + win])
        ys.append(class_to_i[name])
    if not xs:
        return np.empty((0, win, N_CH), np.float32), np.empty((0,), np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def global_mask(sessions: list[Session], cfg: dict) -> list[bool]:
    """True = keep. A channel is removed only when hard-fail is common across files."""
    if not sessions or not cfg.get("auto_drop_hard_failures", True):
        mask = [True] * N_CH
    else:
        frac = float(cfg.get("global_drop_fraction", 0.5))
        fails = np.mean([[1.0 if f else 0.0 for f in s.hard_fail] for s in sessions], axis=0)
        mask = [bool(v < frac) for v in fails]
    for ch in cfg.get("drop_channels") or []:
        idx = int(ch) - 1
        if 0 <= idx < N_CH:
            mask[idx] = False
    if not any(mask):
        raise SystemExit("Channel QC dropped every channel. Clear drop_channels or check the recordings.")
    return mask


def apply_mask(windows: np.ndarray, mask: list[bool]) -> np.ndarray:
    if windows.size == 0:
        return windows
    out = windows.copy()
    for i, keep in enumerate(mask):
        if not keep:
            out[:, :, i] = 0.0
    return out
