# Research log


## Quick review

Ship classifier is [20260927_092039](runs/20260927_092039/report.md): val 74.2%, test **60.5%**. Not a training failure. Held-out sessions disagree.

| Q | Result | Next |
| --- | --- | --- |
| 1. One bad test session? | No. 5=77.9%, 12=57.8%, 15=43.3%. Drop-15 still 67.8%. | Session shift, not more epochs. |
| 2. LOSO variance? | **Not run.** 12 folds x 32 epochs on CPU is too slow. | GPU/SSH. |
| 3. Is pinch learnable? | At 1 s, no. Model pinch MAE 0.205 vs median-0 MAE 0.176; r=0.015. | Target is sparse (median 0, 70% held). |
| 4. 0.5 s pose window? | Pinch starts: r 0.29, MAE 0.172. Overall MAE 0.122 vs 0.126. | Keep 0.5 s for pose. |
| 5. Session-balanced sampler? | **Fail.** Test 48.1% vs 60.5%. Stopped epoch 8. | Do not ship. |

## Comparison

Classifier, same split (seed 7: train 1,2,13,7,9,11,14,4,6 / val 3,10,8 / test 12,5,15):

| Run | Command | Epoch | Val | Test | Verdict |
| --- | --- | --- | --- | --- | --- |
| [092039](runs/20260927_092039/report.md) | `python train.py` | 32/40 | 74.2% | **60.5%** | ship |
| [104411](runs/20260927_104411/report.md) | `--session-balanced` | 8/40 | 71.2% | 48.1% | fail |
| [140200](runs/20260926_140200/report.md) | `python train.py` | 2/40 | 37.3% | 32.1% | fail, superseded |

Pose, same split:

| Run | Window | Epoch | Test MAE | Test r | Pinch MAE | Pinch r | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| median-0 baseline | 1.0 s | - | 0.140 | n/a | 0.176 | n/a | floor |
| [160145](runs/20260926_160145/pose_report.md) | 1.0 s | 3/40 | 0.126 | 0.47 | 0.205 | 0.015 | pinch lose to constant 0 |
| [103801](runs/20260927_103801/pose_report.md) | 0.5 s | 33/40 | **0.122** | **0.52** | **0.172** | **0.29** | pinch weakly learnable |

## Lessons

- 60.5% is session-to-session, not one rotten file. 15 kills fist and point (recall 0). 5 looks strong except rest almost all predicted pinch.
- Do not add classifier epochs. Confusions to fix: rest/open vs pinch, point vs thumb-up, plus session shift.
- Session-balanced sampling hurt generalization. Logged as a failure.
- Pinch at 1 s is a target problem (train median 0, lag-1 held ~70%). 0.5 s is the first setting where pinch beats the naive baseline.
- LOSO stays blocked until GPU. Ask; do not grind 12 folds on CPU.

## How to add a run

1. Prepend a row to the index.
2. Same protocol, different seed/epoch: stop at the row.
3. If it changes what we train next, or it failed: short block under Runs.
4. Update this log in the same turn as the run. Failures too.

## Protocol (current `config.yaml`)

Note only deltas in run rows.

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
| 2026-09-27 | [20260927_104411](runs/20260927_104411/report.md) | clf | `python train.py --session-balanced` | 8/40 | 71.2% | 48.1% | Fail. Equalizing sessions hurt test. |
| 2026-09-27 | [20260927_103801](runs/20260927_103801/pose_report.md) | pose | `python train.py --pose --window-sec 0.5` | 33/40 | MAE 0.092 | MAE 0.122, r 0.52 | Pinch r 0.29. Keep 0.5 s. |
| 2026-09-27 | [20260927_092039](runs/20260927_092039/per_test_session.json) | clf score | no retrain; per-test-session | 32 | - | 12=57.8 / 5=77.9 / 15=43.3 | 60.5% is not one bad session. |
| 2026-09-27 | [20260926_160145](runs/20260926_160145/pose_baseline.json) | pose score | median baseline, no retrain | - | - | MAE 0.140; pinch 0.176 | 1 s model loses to constant 0 on pinch. |
| 2026-09-27 | - | clf LOSO | `python train.py --loso` | - | - | - | Skipped. CPU too slow. Need GPU. |
| 2026-09-27 | [20260927_092039](runs/20260927_092039/report.md) | clf | `python train.py` | 32/40 | 74.2% | 60.5% | Learns. rest/open vs pinch; point vs thumb-up. |
| 2026-09-26 | [20260926_160145](runs/20260926_160145/pose_report.md) | pose | `python train.py --pose` | 3/40 | MAE 0.101 | MAE 0.126, r 0.47 | Pinch r ~ 0. Stopped early. |
| 2026-09-26 | [20260926_140200](runs/20260926_140200/report.md) | clf | `python train.py` | 2/40 | 37.3% | 32.1% | Epoch-2 stop. Superseded. |

Shared split (seed 7): train 1,2,13,7,9,11,14,4,6 / val 3,10,8 / test 12,5,15. Channel mask: all 8 kept.

## Runs

### 20260927_104411 classifier (fail)

Command: `python train.py --session-balanced`. Device: cpu. Epoch 8.

Val 71.2%, test 48.1%. Same split and window counts as 092039.

Test confusion (rows true, order rest fist open pinch point thumb-up):

```
rest      143    0    3  115    1    0
fist       21  105   50   17    4   48
open       12    0  143  150    0    0
pinch      73    1    2  223    1    5
point     112    0    9    0  120   36
thumb-up  121    0    7    0   77   78
```

Decision: do not use session-balanced sampling.

### 20260927_103801 pose 0.5 s

Command: `python train.py --pose --window-sec 0.5`. Epoch 33. Windows 11138 / 3918 / 3420.

| DoF | MAE | r |
| --- | --- | --- |
| thumb_curl | 0.046 | 0.56 |
| index_curl | 0.162 | 0.37 |
| middle_curl | 0.135 | 0.65 |
| ring_curl | 0.143 | 0.57 |
| pinky_curl | 0.082 | 0.57 |
| pinch | 0.172 | 0.29 |
| openness | 0.117 | 0.59 |

Decision: 0.5 s is better for pinch. Index r got worse (0.37 vs 0.44). Overall still a small win.

### 20260927_092039 per-test-session (no retrain)

From [per_test_session.json](runs/20260927_092039/per_test_session.json). Pooled test 60.5% (n=1677).

| Session | n | bal-acc | What breaks |
| --- | --- | --- | --- |
| 5 | 583 | 77.9% | rest 1/96, almost all to pinch |
| 12 | 558 | 57.8% | open->pinch, point->thumb-up |
| 15 | 536 | 43.3% | fist 0/67, point 0/88 |

Drop-one remaining bal-acc: drop 15 = 67.8%, drop 12 = 61.8%, drop 5 = 51.3%.

Decision: 60.5% is mixed session shift, not one outlier.

### 20260926_160145 pose + median baseline

Command: `python train.py --pose`. Windows 11113 / 3932 / 3402. Epoch 3. Test MAE 0.126, r 0.47.

Pinch train: median 0, mean 0.18, std 0.26, lag-1 held 0.70. Other DoFs held ~0.36-0.41.

Naive train-median on test ([pose_baseline.json](runs/20260926_160145/pose_baseline.json)): MAE 0.140. Pinch MAE 0.176 (constant 0). Model pinch MAE 0.205, r 0.015: worse than the constant.

Decision: at 1 s, pinch is not a model-capacity problem.

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

Decision: do not add epochs. Head/loss on rest-pinch, open-pinch, point-thumb-up, after LOSO on GPU.

### 20260926_140200 classifier (fail)

Same split and window counts as 092039. Kept epoch 2. Test mass on pinch. Superseded.

