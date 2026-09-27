# Research log

Notebook. Cite `runs/<timestamp>/`. Newest at the top of each list.

## Quick review

Classifier [20260927_092039](runs/20260927_092039/report.md): val 74.2%, pooled test 60.5% (n=1677). Val-test gap is not explained by under-training (best checkpoint epoch 32/40). Held-out sessions are heterogeneous.

Current pose baseline is 0.5 s, one seed: [20260927_103801](runs/20260927_103801/pose_report.md).

| Q | Measurement | Interpretation |
| --- | --- | --- |
| 1. One or two bad test sessions? | 5=77.9% (n=583), 12=57.8% (n=558), 15=43.3% (n=536). Drop-15 remaining 67.8%. Failure modes differ by session. | Large held-out-session heterogeneity. EMG session shift is consistent with this, not isolated from label mix, recording quality, or optical/guided timing. LOSO + per-session QC needed. |
| 2. LOSO variance? | Not run. 12 folds x 32 epochs on CPU. | Blocked. GPU/SSH. |
| 3. Pinch under 1 s formulation? | Model pinch MAE 0.205, r=0.015 vs constant-0 MAE 0.176. Train pinch: median 0, lag-1 held ~70%. | Current 1 s formulation does not learn pinch better than the constant baseline. |
| 4. 0.5 s pose window? | One run. MAE 0.126 -> 0.122, r 0.47 -> 0.52, pinch r 0.015 -> 0.29, pinch MAE 0.176 (const 0) -> 0.172. Index r 0.44 -> 0.37. | Pinch is weakly learnable at 0.5 s. Consistent with 1 s aggregation smearing a dynamic target. Not a multi-seed result. |
| 5. This session-balanced sampler? | Test 48.1% vs 60.5%. Stopped epoch 8 (baseline best=32). Thumb-up recall 75% -> 28%, fist 58% -> 43%, rest 42% -> 55%. | Failed implementation. Do not use this sampler. Does not show that session balancing is generally harmful. |

## Comparison

Classifier, same split (seed 7: train 1,2,13,7,9,11,14,4,6 / val 3,10,8 / test 12,5,15):

| Run | Command | Epoch | Val | Test | Note |
| --- | --- | --- | --- | --- | --- |
| [092039](runs/20260927_092039/report.md) | `python train.py` | 32/40 | 74.2% | 60.5% | current clf checkpoint |
| [104411](runs/20260927_104411/report.md) | `--session-balanced` | 8/40 | 71.2% | 48.1% | this sampler; not comparable epoch |
| [140200](runs/20260926_140200/report.md) | `python train.py` | 2/40 | 37.3% | 32.1% | early stop; superseded |

Pose, same split, hop 0.1 s:

| Run | Window | Epoch | Test MAE | Test r | Pinch MAE | Pinch r | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| median / const-0 | 1.0 s | - | 0.140 | n/a | 0.176 | n/a | naive baseline |
| [160145](runs/20260926_160145/pose_report.md) | 1.0 s | 3/40 | 0.126 | 0.47 | 0.205 | 0.015 | pinch worse than const-0 |
| [103801](runs/20260927_103801/pose_report.md) | 0.5 s | 33/40 | 0.122 | 0.52 | 0.172 | 0.29 | current pose baseline; one seed |

Pose baseline (freeze except the next one-factor change): window 0.5 s, hop 0.1 s, same CNN, same loss, same split. Test MAE 0.122, r 0.52, pinch MAE 0.172, pinch r 0.29.

## Lessons

- Classifier pooled 60.5% is an average over dissimilar held-out recordings, not one outlier file. Session 15: fist 0/67, point 0/88. Session 5: rest 1/96, mostly predicted pinch. Session 12: open->pinch, point->thumb-up. Cause (EMG shift vs prevalence vs QC vs timing) is not identified.
- Freeze clf architecture, loss, and sampling until LOSO (GPU) and a per-session QC/prevalence table. Adding epochs, this sampler, or a new loss is not justified by the current evidence.
- `--session-balanced` as implemented changed decision boundaries and stopped at epoch 8. Logged as a failed run of this sampler.
- Pinch at 1 s is primarily a target/windowing problem under the current formulation: sparse target, ~70% held, and the model loses to the constant baseline. At 0.5 s pinch is weakly learnable (r=0.29; MAE 0.172 vs 0.176). Not solved.
- Next pose factor: 0.25 s vs 0.5 s, same everything else. Temporal optimum before GRUs, multitask, or pretraining.
- LOSO not run on CPU. Ask for GPU.

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
| 2026-09-27 | [20260927_104411](runs/20260927_104411/report.md) | clf | `python train.py --session-balanced` | 8/40 | 71.2% | 48.1% | This sampler. Epoch 8 vs 32. Thumb-up recall 75%->28%. |
| 2026-09-27 | [20260927_103801](runs/20260927_103801/pose_report.md) | pose | `python train.py --pose --window-sec 0.5` | 33/40 | MAE 0.092 | MAE 0.122, r 0.52 | Pinch r 0.29. New pose baseline; one seed. |
| 2026-09-27 | [20260927_092039](runs/20260927_092039/per_test_session.json) | clf score | no retrain; per-test-session | 32 | - | 12=57.8 / 5=77.9 / 15=43.3 | Held-out sessions heterogeneous; not one file. |
| 2026-09-27 | [20260926_160145](runs/20260926_160145/pose_baseline.json) | pose score | median / const-0, no retrain | - | - | MAE 0.140; pinch 0.176 | 1 s model pinch MAE 0.205, worse than const-0. |
| 2026-09-27 | - | clf LOSO | `python train.py --loso` | - | - | - | Not run. CPU. Need GPU. |
| 2026-09-27 | [20260927_092039](runs/20260927_092039/report.md) | clf | `python train.py` | 32/40 | 74.2% | 60.5% | Current clf. rest/open vs pinch; point vs thumb-up. |
| 2026-09-26 | [20260926_160145](runs/20260926_160145/pose_report.md) | pose | `python train.py --pose` | 3/40 | MAE 0.101 | MAE 0.126, r 0.47 | Pinch r 0.015. Early stop. |
| 2026-09-26 | [20260926_140200](runs/20260926_140200/report.md) | clf | `python train.py` | 2/40 | 37.3% | 32.1% | Epoch-2 stop. Superseded. |

Shared split (seed 7): train 1,2,13,7,9,11,14,4,6 / val 3,10,8 / test 12,5,15. Channel mask: all 8 kept.

## Runs

### 20260927_104411 classifier (this sampler)

Command: `python train.py --session-balanced`. Device: cpu. Epoch kept 8/40 (patience). Baseline best checkpoint was epoch 32.

Val 71.2%, test 48.1%. Same split and window counts as 092039.

Test recall vs 092039: rest 42% -> 55%, fist 58% -> 43%, open 55% -> 47%, pinch 92% -> 73%, point 41% -> 43%, thumb-up 75% -> 28%.

Test confusion (rows true, order rest fist open pinch point thumb-up):

```
rest      143    0    3  115    1    0
fist       21  105   50   17    4   48
open       12    0  143  150    0    0
pinch      73    1    2  223    1    5
point     112    0    9    0  120   36
thumb-up  121    0    7    0   77   78
```

Do not reuse this sampler. Not evidence that session balancing in general is harmful.

### 20260927_103801 pose 0.5 s (current pose baseline)

Command: `python train.py --pose --window-sec 0.5`. Epoch 33. Windows 11138 / 3918 / 3420. One seed.

| DoF | MAE | r |
| --- | --- | --- |
| thumb_curl | 0.046 | 0.56 |
| index_curl | 0.162 | 0.37 |
| middle_curl | 0.135 | 0.65 |
| ring_curl | 0.143 | 0.57 |
| pinky_curl | 0.082 | 0.57 |
| pinch | 0.172 | 0.29 |
| openness | 0.117 | 0.59 |

Vs 1.0 s: overall MAE 0.126 -> 0.122, r 0.47 -> 0.52. Pinch r 0.015 -> 0.29; pinch MAE vs const-0 0.176 -> 0.172. Index r 0.44 -> 0.37.

Hypothesis: 1 s window averages away pinch-related transients. Next factor: 0.25 s vs 0.5 s.

### 20260927_092039 per-test-session (no retrain)

From [per_test_session.json](runs/20260927_092039/per_test_session.json). Pooled test 60.5% (n=1677).

| Session | n | bal-acc | Errors |
| --- | --- | --- | --- |
| 5 | 583 | 77.9% | rest 1/96, almost all to pinch |
| 12 | 558 | 57.8% | open->pinch, point->thumb-up |
| 15 | 536 | 43.3% | fist 0/67, point 0/88 |

Drop-one remaining bal-acc: drop 15 = 67.8%, drop 12 = 61.8%, drop 5 = 51.3%.

Held-out-session heterogeneity, with different error modes. EMG session shift is one hypothesis. Alternatives: class prevalence, recording/QC, optical or guided timing. Not identified here.

### 20260926_160145 pose 1.0 s + median / const-0 baseline

Command: `python train.py --pose`. Windows 11113 / 3932 / 3402. Epoch 3. Test MAE 0.126, r 0.47.

Pinch train: median 0, mean 0.18, std 0.26, lag-1 held 0.70. Other DoFs held ~0.36-0.41.

Naive train-median on test ([pose_baseline.json](runs/20260926_160145/pose_baseline.json)): MAE 0.140. Pinch MAE 0.176 (constant 0). Model pinch MAE 0.205, r 0.015.

The current 1 s formulation does not learn pinch better than the constant baseline. Capacity is not the measured bottleneck for pinch at 1 s.

### 20260927_092039 classifier (current clf checkpoint)

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

Pooled errors: rest/open vs pinch, point vs thumb-up. Do not add epochs. No architecture/loss/sampler change until LOSO and per-session QC.

### 20260926_140200 classifier

Same split and window counts as 092039. Kept epoch 2. Test mass on pinch. Superseded.

