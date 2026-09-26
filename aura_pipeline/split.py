from __future__ import annotations

import numpy as np


def assign_sessions(names: list[str], cfg: dict) -> dict[str, list[str]]:
    """Assign each session file to train, validation, or test.

    Windows from one file stay in one role. Adjacent windows overlap.
    """
    known = set(names)
    pinned = {
        "train": list(cfg.get("train_sessions") or []),
        "val": list(cfg.get("val_sessions") or []),
        "test": list(cfg.get("test_sessions") or []),
    }
    if any(pinned.values()):
        missing = [n for group in pinned.values() for n in group if n not in known]
        if missing:
            raise SystemExit("Split lists name unknown sessions: " + ", ".join(missing))
        used = [n for group in pinned.values() for n in group]
        if len(used) != len(set(used)):
            raise SystemExit("A session is listed in more than one of train, val, test.")
        leftover = [n for n in names if n not in set(used)]
        rng = np.random.default_rng(int(cfg.get("seed", 7)))
        rng.shuffle(leftover)
        if not pinned["test"]:
            n_test = max(1, int(round(len(names) * float(cfg.get("test_fraction", 0.2)))))
            pinned["test"] = leftover[:n_test]
            leftover = leftover[n_test:]
        if not pinned["val"]:
            n_val = max(1, int(round(len(names) * float(cfg.get("val_fraction", 0.2)))))
            pinned["val"] = leftover[:n_val]
            leftover = leftover[n_val:]
        pinned["train"] = pinned["train"] + leftover
        if not pinned["train"] or not pinned["val"] or not pinned["test"]:
            raise SystemExit("Split needs at least one train, one val, and one test session.")
        return pinned

    if len(names) < 3:
        raise SystemExit(
            f"Need at least 3 session files for train / val / test. Found {len(names)}. "
            "Add recordings to data/raw."
        )

    order = list(names)
    rng = np.random.default_rng(int(cfg.get("seed", 7)))
    rng.shuffle(order)
    n_test = max(1, int(round(len(order) * float(cfg.get("test_fraction", 0.2)))))
    n_val = max(1, int(round(len(order) * float(cfg.get("val_fraction", 0.2)))))
    if n_test + n_val >= len(order):
        n_test, n_val = 1, 1
    return {
        "test": order[:n_test],
        "val": order[n_test:n_test + n_val],
        "train": order[n_test + n_val:],
    }
