# Research log

Training notebook. One row in the index per run. One short block under Runs only when the result needs a confusion matrix, per-DoF table, or a decision. Do not write an essay for a single experiment.

Cite `runs/<timestamp>/`. Numbers come from that folder, not from memory. Newest run at the top of each list.

## How to add a run

1. Prepend a row to the index.
2. If the run is just "same protocol, different seed/epoch", stop there.
3. If the finding changes what we train next, add a short block under Runs (command, split, scores, matrix or per-DoF, one-line decision).
4. Do not paste epoch traces unless early-stop vs full budget is the finding.

```
| YYYY-MM-DD | timestamp | clf or pose | command | epoch kept | val | test | one-line finding |
```

## Protocol (current `config.yaml`)

Do not copy this into every run. Note only deltas.

- 250 Hz, window 1.0 s, hop 0.1 s, edge trim 0.2 s, min label fraction 0.9
- Causal bandpass 20-120 Hz, bandstop 45-55 Hz, notch 100 Hz. Filter and scale outside ONNX
- Split unit: session file. Empty lists + seed 7 + 20% val / 20% test
- Classifier: AuraNet, class-weighted CE, AdamW, 40 epochs, patience 8, batch 64, lr 1e-3
- Pose: same backbone, sigmoid 7-DoF, weighted Smooth L1, early stop on val MAE
- Classes: rest, fist, open, pinch, point, thumb-up
- Pose DoF: thumb_curl, index_curl, middle_curl, ring_curl, pinky_curl, pinch, openness

## Index

| Date | Run | Task | Command | Epoch | Val | Test | Finding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-27 | [20260927_092039](runs/20260927_092039/report.md) | clf | `python train.py` | 32/40 | 74.2% bal-acc | 60.5% bal-acc | Learns. Errors are rest/open vs pinch and point vs thumb-up. |
| 2026-09-26 | [20260926_160145](runs/20260926_160145/pose_report.md) | pose | `python train.py --pose` | 3/40 | MAE 0.1008 | MAE 0.1260, r 0.47 | Pinch r ~ 0. Stopped early. |
| 2026-09-26 | [20260926_140200](runs/20260926_140200/report.md) | clf | `python train.py` | 2/40 | 37.3% bal-acc | 32.1% bal-acc | Same split as 092039. Epoch-2 stop, not a trained model. |

Shared split (seed 7): train 1,2,13,7,9,11,14,4,6 / val 3,10,8 / test 12,5,15. Channel mask: all 8 kept.

## Runs

### 20260927_092039 classifier

Command: `python train.py`. Device: cpu. Train windows 4449 (rest 580, fist 718, open 816, pinch 825, point 734, thumb-up 776).

Val 0.24 (ep 1) -> 0.37 (ep 2) -> 0.74 (ep 32). Loss still falling at 40. Patience never fired.

Test confusion (rows true, order rest fist open pinch point thumb-up):

```
rest      110    0    5  145    0    2
fist        8  143    3   16    0   75
open        1    0  168  136    0    0
pinch      16    0    5  279    0    5
point      23    0    0    7  114  133
thumb-up   51    0    0    1   19  212
```

Decision: do not add epochs. Next change is the head/loss on rest-pinch, open-pinch, point-thumb-up.

### 20260926_160145 pose

Command: `python train.py --pose`. Windows 11113 / 3932 / 3402. Test pinch r = 0.0145; other DoF r 0.44-0.64. Epoch 3.

### 20260926_140200 classifier

Same split and window counts as 092039. Kept epoch 2. Test mass on pinch. Superseded.
