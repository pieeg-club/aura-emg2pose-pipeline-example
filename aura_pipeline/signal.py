from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, resample, sosfilt, sosfilt_zi, tf2sos


def design_sos(fs: float, cfg: dict) -> np.ndarray:
    """Causal SOS: bandpass, mains bandstop, harmonic notch. Coefficients come from config."""
    lo, hi = cfg["bandpass_hz"]
    order = int(cfg["bandpass_order"])
    mains = float(cfg["notch_mains_hz"])
    parts = [
        butter(order, (lo, hi), btype="bandpass", fs=fs, output="sos"),
        butter(order, (mains - 5.0, mains + 5.0), btype="bandstop", fs=fs, output="sos"),
        tf2sos(*iirnotch(float(cfg["notch_harmonic_hz"]), Q=float(cfg["notch_q"]), fs=fs)),
    ]
    return np.vstack(parts)


def causal_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Forward SOS filter. Initial state is set from the first sample."""
    if len(x) == 0:
        return x.astype(np.float32)
    zi = np.repeat(sosfilt_zi(sos)[:, :, None], x.shape[1], axis=2)
    zi = zi * x[0].astype(np.float64)[None, None, :]
    y, _ = sosfilt(sos, x.astype(np.float64), axis=0, zi=zi)
    return y.astype(np.float32)


def to_rate(x: np.ndarray, t: np.ndarray, target_fs: float) -> tuple[np.ndarray, float, int | None]:
    """Return signal, sample rate, and new length if resampled."""
    if len(t) < 3:
        return x, target_fs, None
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return x, target_fs, None
    fs = 1.0 / dt
    if abs(fs - target_fs) / target_fs < 0.02:
        return x, fs, None
    n = int(round(len(x) * target_fs / fs))
    if n < 8:
        return x, fs, None
    y = resample(x, n, axis=0).astype(np.float32)
    return y, target_fs, n


def robust_scale(x: np.ndarray) -> np.ndarray:
    scale = np.median(np.abs(x), axis=0) / 0.6745
    scale = np.asarray(scale, dtype=np.float32)
    scale[scale < 1e-6] = 1.0
    return scale


def mains_ratio(x: np.ndarray, fs: float, mains: float = 50.0) -> np.ndarray:
    """Fraction of 20-120 Hz power sitting in a 2 Hz band around mains. Per channel."""
    n = x.shape[0]
    if n < int(fs):
        return np.full(x.shape[1], np.nan, dtype=np.float32)
    spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= 20.0) & (freqs <= 120.0)
    line = (freqs >= mains - 1.0) & (freqs <= mains + 1.0)
    total = spec[band].sum(axis=0) + 1e-12
    return (spec[line].sum(axis=0) / total).astype(np.float32)
