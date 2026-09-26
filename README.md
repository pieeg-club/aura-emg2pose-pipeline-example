# Aura hand EMG

Baseline pipeline for the Aura hand-EMG Kaggle competition. Aura is an 8-channel surface-EMG device worn on the palm, and the competition dataset is recorded from it. One session file goes in and one ONNX model comes out. Two targets are trained from the same recordings:

- Gesture classifier: six classes, rest, fist, open, pinch, point, thumb-up (`python train.py`).
- Hand-pose regressor: seven continuous DoF, the five per-finger curls plus pinch and openness (`python train.py --pose`).

This directory is the working tree. Run every command from here.

## Data

Place one file per session in `data/raw/`. Accepted formats are `.xlsx` (first sheet) and `.csv`. Excel lock files (`~$*`) are ignored.

| Column | Role |
| --- | --- |
| `t` | Time in seconds from session start |
| `emg_new` | If present, only rows with value 1 are kept |
| `ch1` … `ch8` | EMG in microvolts |
| `label` | Class name |
| `label_source` | If present, only `guided` rows are labeled. Other rows are unlabeled |
| `optical_*` | Optical tracker metadata. Not a classifier input. See `drop_optical_disagreement` |
| `thumb_curl` … `pinky_curl`, `pinch`, `openness` | Continuous pose in `[0, 1]`. The `--pose` target, not an input |

IMU columns (`ax`, `ay`, `az`, `gx`, `gy`, `gz`, `roll`, `pitch`, `yaw`) are not read. Class names are normalized (`open hand` → `open`, `thumb up` → `thumb-up`). Anything outside the six-class list is unlabeled.

Recordings are not part of the repository. `data/raw/` is gitignored except for its note.

If `data/raw/` is empty, `fallback_data_dir` in `config.yaml` is used when that path exists.

## Method

Sampling rate is 250 Hz. Files more than 2% off that rate are resampled. The first 0.5 s after filtering is dropped as filter warmup.

Preprocessing, applied before the network and not inside the ONNX graph:

1. Causal bandpass 20–120 Hz, order 4.
2. Causal bandstop 45–55 Hz, order 4.
3. Causal notch at 100 Hz, Q = 30.
4. Per-channel scale `median(|x|) / 0.6745`, with a floor of `1e-6`.

Windows are 1.0 s (250 samples) with a hop of 0.1 s. Samples within 0.2 s of a label change are unlabeled, so ramp intervals are not used as holds. A window is kept when at least 90% of its samples share one class and no second class is present.

Channel QC, per file, on the filtered signal:

- Drop if the channel is empty, flat (std < 1 µV or peak-to-peak < 2 µV), railed, or listed in `drop_channels`.
- Report SNR of active grips versus rest, and the fraction of 20–120 Hz power in the 49–51 Hz band. Low SNR is not an automatic drop.
- A channel is zeroed in the shipped graph only if it hard-fails in at least `global_drop_fraction` of the train files, or it is listed in `drop_channels`. The mask is estimated from the train files only, so neither validation nor test decides which channels exist.

The network is a temporal-spatial convolutional classifier (time convolution, depthwise mix across the 8 channels, separable convolution, linear head). Input layout is `(batch, time, channels)`. Output is a softmax over the six classes, index 0 = rest. Training uses class-weighted cross-entropy and AdamW. Early stopping watches validation balanced accuracy.

## Pose regression

`python train.py --pose` trains a continuous hand-pose regressor on the same recordings and the same windows. The target is the optical hand tracker carried in each session file: seven values in `[0, 1]`, the five per-finger curls plus pinch and openness. This is an optical estimate, not motion capture, so the model distils that tracker into EMG and its ceiling is the tracker's own quality.

The target for a window is the confidence-weighted median pose over that window. A window is used when at least half its samples carry a present, valid pose and the median optical confidence clears `pose_min_conf`. Loss is Smooth L1 weighted by confidence, and early stopping watches validation mean absolute error. The network reuses the classifier backbone with a sigmoid head, so the output is a pose vector in `[0, 1]`. Reported metrics are per-DoF MAE and Pearson correlation on the test sessions.

## Split

The split unit is the session file. Windows from one file are not placed on both sides of a split: the hop is 100 ms inside a 1 s window, so neighboring windows are the same contraction.

With the session lists in `config.yaml` empty, files are shuffled with `seed` and cut by `test_fraction` and `val_fraction` (default 20% test, 20% validation, remainder train). At least one file is required in each role, and at least three files are required in total.

| Role | Use |
| --- | --- |
| Train | Weight updates |
| Validation | Early stopping only |
| Test | One score after training. Not in the loss, not in early stopping, not in the exported weights |

`runs/<timestamp>/split.json` is the assignment from that run. Copy it into `train_sessions`, `val_sessions`, and `test_sessions` to freeze the split. Files that are not listed are appended to train.

`--loso` runs leave-one-session-out on train and validation only. It does not score or modify the test files, and it does not replace `model.onnx`.

The reported test metric is balanced accuracy. Rest is the majority class, so unweighted accuracy is not the selection metric.

## Setup

Python 3.12.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencies: NumPy, pandas, openpyxl, SciPy, PyYAML, PyTorch, ONNX, ONNX Runtime.

## Commands

```powershell
python train.py --qc-only
python train.py
python train.py --pose
python train.py --data D:\more-sessions
python train.py --test multimodal_15
python train.py --loso
python train.py --self-test
python infer.py
python infer.py runs\RUN\model.onnx
```

`--qc-only` writes the channel table and does not train. `--pose` trains the hand-pose regressor instead of the classifier and writes `pose.onnx` beside the other artifacts. `--self-test` trains for one epoch on synthetic tensors and checks that the ONNX file loads. It does not read `data/raw/`.

`infer.py` is the classifier caller. It loads `model.onnx`, runs a mock 1 s buffer through the same causal filter and per-channel scale as training, and prints class probabilities. Filter, scale, and the 0.5 s warmup are not in the graph. Copy this file; do not feed raw microvolts to ONNX. With no path it uses `runs/latest.txt` if that run has `model.onnx`, otherwise the newest `runs/*/model.onnx`.

## Run artifacts

Each training writes `runs/<timestamp>/`. `runs/latest.txt` stores that path.

| File | Contents |
| --- | --- |
| `split.json` | Train, validation, and test session names |
| `qc.md`, `qc.json` | Per-session channel std, SNR, mains ratio, keep/drop |
| `report.md`, `report.json` | Train counts, validation score, test score, test confusion matrix |
| `model.onnx` | Exported network. Weights from the train split only |
| `model.pt` | PyTorch checkpoint, same weights, plus the split and channel mask |
| `contract.json` | Inference contract below |
| `pose.onnx` | Pose regressor. Present only after `--pose` |
| `pose.pt` | Pose checkpoint, plus DoF names, split, and channel mask. `--pose` only |
| `pose_contract.json`, `pose_report.md`, `pose_report.json` | Pose contract and scores. `--pose` only |
| `loso.json` | Present only after `--loso` |

## Inference contract

`contract.json` is authoritative for a given run. Defaults:

| Field | Value |
| --- | --- |
| Input name | `emg` |
| Input | `float32`, `[batch, 250, 8]`, time then channel |
| Output name | `probs` |
| Output | `float32`, `[batch, 6]`, sums to 1 |
| Class order | `rest`, `fist`, `open`, `pinch`, `point`, `thumb-up` |
| Sample rate | 250 Hz |
| Graph contents | Classifier only. Filter, scale, and warmup are the caller's responsibility |
| Channel mask | Zeroed inside the graph. See `channel_keep` |

```powershell
python infer.py
python infer.py runs\RUN\model.onnx
```

The mock buffer is synthetic EMG in microvolts. Scale is computed on that 1 s window; training uses the full recording. A live caller should keep SOS filter state across hops instead of re-applying the 0.5 s warmup every window.

## Pose inference contract

`pose_contract.json` is authoritative for a `--pose` run. Defaults:

| Field | Value |
| --- | --- |
| Input | `float32`, `[batch, 250, 8]`, time then channel |
| Output name | `pose` |
| Output | `float32`, `[batch, 7]`, each value in `[0, 1]` |
| DoF order | `thumb_curl`, `index_curl`, `middle_curl`, `ring_curl`, `pinky_curl`, `pinch`, `openness` |
| Target source | Optical hand tracker, not motion capture |
| Preprocess | Same causal filter and per-channel scale as the classifier, applied by the caller |

## Layout

```
config.yaml          training and split configuration
train.py             entry point
infer.py             one-window ONNX caller (filter and scale outside the graph)
aura_pipeline/       load, QC, windows, model, fit, export, pose
data/raw/            session files
runs/                outputs, gitignored
```

## Configuration

All knobs are in `config.yaml`. Do not hard-code channels, epochs, or split lists in Python.

`drop_optical_disagreement` defaults to false. Optical fields are last-value holds on the EMG time base, so a sample-level veto is not a reliable label. When enabled, a guided sample is unlabeled if `optical_present` is set, `optical_conf` is at least `optical_conf_min`, and `optical_class` names a different class. The pair rest/open is exempt.

`pose_targets`, `pose_min_conf`, and `pose_loss_beta` configure the `--pose` regressor: the DoF columns to predict, the median-confidence floor for keeping a window, and the Smooth L1 transition point.
