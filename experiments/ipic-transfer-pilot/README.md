# IPIC four-arm transfer pilot: verified result — November 4, 2026

**All four arms completed; the prespecified scientific gate failed.** Each arm saved exactly 200 updates, 800 total. Training and evaluation finished approximately 18:57 KST; independent verification completed at 2026-11-04 T19:06:01.957075+09:00. Controller and all four containers exited successfully without OOM or guard stops.

The comparison uses 23 reviewed development frames (355 visible filled-sack points) from two recording groups:18 positive frames and 5 empty frames. Three low-count frames contain 1–11 sacks. Maximum cohort count is 36; this is not FSC dense/extreme evaluation.

## Primary frozen threshold: 0.50

| Arm | Change | MAE ↓ | RMSE ↓ | F1@8 on positive images ↑ | GAME1 / GAME2 ↓ |
|---|---|---:|---:|---:|---:|
| P00 | Original coverage + current appearance | 15.4348 | 19.4254 | 0.0000 | 15.4348 / 15.4348 |
| P10 | Broader coverage + current appearance | 15.4348 | 19.4254 | 0.0000 | 15.4348 / 15.4348 |
| P01 | Original coverage + gentler appearance | 15.4348 | 19.4254 | 0.0000 | 15.4348 / 15.4348 |
| P11 | Broader coverage + gentler appearance (primary) | 15.4348 | 19.4254 | 0.0000 | 15.4348 / 15.4348 |

All four predict zero sacks on all 23 frames:0 true positives, 0 false positives and 355 false negatives. The raw models are different: all four checkpoint and final raw-prediction hashes differ, and maximum probabilities range 0.3277–0.3849. Identical reported metrics arise after the fixed threshold, not from reusing the same output.

The prespecified all-image macro F1@8 is 0.217391 in every arm because five empty/empty cases receive F1=1. Pooled F1 and positive-only macro F1 are both 0. This empty-image credit is not localization success.

P11 fails the required 10% MAE reduction and 0.05 macro-F1 increase against each of P00, P10, P01: every primary signed difference is 0. Protection checks formally pass because outputs are identical. Low-count MAE/GAME1/GAME2 are 3.6667 and F1=0; higher-count MAE is 22.9333; empty-frame false positives are 0. Thus there is no gain without harm to claim. Both development groups fail to produce detections: MAE 17.9333 on 15 March 12 frames and 10.7500 on 8 February 27 frames.

## Fixed diagnostic threshold: 0.10

This threshold was prespecified, not selected after results, and cannot rescue the primary gate.

| Arm | MAE ↓ | Macro F1@8 ↑ | GAME1 / GAME2 ↓ | Low-count MAE ↓ | Empty-frame FP ↓ |
|---|---:|---:|---:|---:|---:|
| P00 | 14.0870 | 0.0465 | 20.7826 / 26.0000 | 6.6667 | 40 |
| P10 | 101.2609 | 0.0510 | 101.4348 / 105.3478 | 70.6667 | 101 |
| P01 | 16.9130 | 0.0518 | 21.2609 / 27.6087 | 13.6667 | 45 |
| P11 | 117.0870 | 0.0492 | 118.0435 / 121.8696 | 81.3333 | 128 |

P11 predicts 3048 points for 355 true sacks, with 114 matched true positives, 2934 false positives and 241 false negatives at the fixed localization tolerance. More detected points do not translate into reliable counting or spatial accuracy. The secondary diagnostic fails MAE and F1 gain requirements, overall RMSE/GAME1/GAME2 protections, low-count MAE/GAME1/GAME2 protections, and empty-FP protection against every control. Low-count F1 alone improves slightly but does not compensate.

| P11 minus control at 0.10 | MAE | RMSE | GAME1 | GAME2 | Low-count MAE | Empty FP |
|---|---:|---:|---:|---:|---:|---:|
| P00 | +103.0000 | +161.2979 | +97.2609 | +95.8696 | +74.6667 | +88 |
| P10 | +15.8261 | +26.1306 | +16.6087 | +16.5217 | +10.6667 | +27 |
| P01 | +100.1739 | +158.5764 | +96.7826 | +94.2609 | +67.6667 | +83 |

At 0.10, P11 also harms both groups relative to P00: March 12 MAE 99.0 vs 15.0, February 27 MAE 151.0 vs 12.375. Conservative appearance alone and broader coverage alone do not supply a defensible successful alternative in this comparison.

## Previously saved pretrained references on the same 23 frames

| System | MAE ↓ | RMSE ↓ | Pooled F1@8 ↑ | Positive-only macro F1@8 ↑ |
|---|---:|---:|---:|---:|
| IPIC: every arm at 0.50 | 15.4348 | 19.4254 | 0.0000 | 0.0000 |
| CountGD++ external prompts | 11.7391 | 14.9870 | 0.1447 | 0.1108 |
| GeCo2 disclosed external adapter | 13.5652 | 17.1794 | 0.0686 | 0.0591 |

Primary IPIC MAE is 3.6957 worse than CountGD++ and 1.8696 worse than the GeCo2 adapter. Reference predictions were reused and their hashes, query coverage and count arithmetic rechecked; no models were rerun. GeCo2 is a disclosed external-reference adapter, not an official native external result. Training supervision, model capacity and compute differ, so this is descriptive performance comparison rather than a controlled causal baseline.

## Verification and decision

Checked 130 frozen source/data files, 41 IPIC source modules, original plan/source digest and saved foundation hash; all 200 sequential update records and 1600 augmentation pairs per arm; exact initial prediction equality; complete unique 23-query coverage and unchanged native truth; finite logs/predictions; final checkpoint hashes and literal metadata. Worker finite model/Adam checks are preserved; the independent collector did not deserialize tensors. MAE/RMSE/bias, point matching and GAME metrics were independently recomputed from saved predictions.

**Do not promote this recipe to longer training.** This bounded coverage-by-appearance intervention did not repair transfer at 200 updates. Its failure does not establish that no longer or different trajectory could learn, and the present outputs do not isolate a unique cause such as calibration, representation or assignment. No threshold search, replacement winner, new seed, additional fit or reserved evaluation has been launched. Preserve all evidence.

These are development results from one seed and two repeatedly used recording groups. They establish neither independent generalization nor a novel counting mechanism or paper readiness.

Machine-readable result · Full verified analysis · Supplemental checks

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 50 files under the folder ipic-transfer-pilot/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
