# IPIC classification repair: verified negative result — November 4, 2026

**All three arms completed. Both the relative mechanism gate and the absolute readiness gate failed. Do not promote this recipe to full training.**

F, U and H each saved all 200 additional updates, reaching cumulative step 400 from the same P11 model and authentic Adam state. Total new optimization was 600 calls. The controller and all containers exited successfully, without OOM or guard stops. Final execution finished approximately **21:12:49 KST**, correcting the DGX host clock by its observed approximate 8-hour-27-second offset; independent verification completed after 21:20 KST. The actual local verification timestamp is preserved in the result receipt.

The fixed development cohort contains 23 reviewed frames, 355 filled-sack points, 18 positive and 5 empty frames from two recording groups. Three frames have 1–11 sacks and 15 have at least 12; the maximum is 36. These are reused development groups, not an untouched test or FSC dense/extreme benchmark.

## Primary threshold 0.50

| Arm | Objective | MAE | RMSE | Positive-frame macro F1@8 | Pooled F1@8 | GAME1 / GAME2 | Predicted points |
|---|---|---:|---:|---:|---:|---:|---:|
| F | Original focal continuation | 15.4348 | 19.4254 | 0.0000 | 0.0000 | 15.4348 / 15.4348 | 0 |
| U | BCE with uniformly sampled negatives | 288.6957 | 374.3046 | 0.0418 | 0.0435 | 288.6957 / 291.5652 | 6995 |
| H, primary | BCE with hard negatives | 15.4348 | 19.4254 | 0.0000 | 0.0000 | 15.4348 / 15.4348 | 0 |

F and H miss all 355 development sacks. Their maximum raw probabilities are 0.4149 and 0.2628 respectively. U raises confidence but produces 160 true positives, **6835 false positives** and 195 false negatives. U's 17/18 positive-frame coverage does not compensate for its severe overcounting. All-image macro F1 is 0.2174 for F/H solely because five empty/empty cases receive F1=1; this is not localization success.

| Primary protection metric | F | U | H |
|---|---:|---:|---:|
| Low-count MAE | 3.6667 | 93.3333 | 3.6667 |
| Low-count F1@8 | 0.0000 | 0.0312 | 0.0000 |
| Low-count GAME1 / GAME2 | 3.6667 / 3.6667 | 93.3333 / 93.3333 | 3.6667 / 3.6667 |
| Higher-count MAE, 12–36 sacks | 22.9333 | 411.8667 | 22.9333 |
| False positives on five empty frames | 0 | 182 | 0 |
| March 12 group MAE, 15 frames | 17.9333 | 284.9333 | 17.9333 |
| February 27 group MAE, 8 frames | 10.7500 | 295.7500 | 10.7500 |

H minus F is exactly zero in primary MAE, positive F1, RMSE, GAME1/2 and low-count metrics. Hence H fails both required gains; its nonregression checks against F pass only because both return empty outputs. Against U, H reduces MAE by 273.2609, RMSE by 354.8791 and GAME1/2 by 273.2609/276.1304, but loses 0.0418 positive-frame macro F1 and 0.0312 low-count F1. Thus it also fails the positive-F1 gain and low-F1 protection against U. Beating an overcounting control by predicting zero does not establish a useful mechanism.

All four absolute requirements fail for H: MAE 15.4348 exceeds 7.7174; pooled F1 is 0 instead of at least 0.50; 0/18 positive frames have a true positive instead of at least 14; fitting positive-frame F1 is 0 instead of at least 0.70.

## Fixed diagnostic threshold 0.10

This threshold was frozen before the test and cannot rescue the primary result.

| Arm | MAE | Positive-frame macro F1@8 | Pooled F1@8 | GAME1 / GAME2 | Low-count MAE | Empty-frame FP |
|---|---:|---:|---:|---:|---:|---:|
| F | 88.8261 | 0.0583 | 0.0684 | 89.7826 / 92.9130 | 73.3333 | 120 |
| U | 879.3913 | 0.0233 | 0.0241 | 879.3913 / 879.5652 | 546.6667 | 890 |
| H | 4154.5217 | 0.0078 | 0.0072 | 4154.5217 / 4154.5217 | 4469.6667 | 4816 |

H selects **95,909 proposals for 355 sacks**, with 346 true positives, 95,563 false positives and 9 false negatives. H minus F/U MAE is **+4065.6957 / +3275.1304**; positive-frame F1 changes are **−0.0505 / −0.0155**. Its RMSE, GAME1/2, low-count MAE/GAME and empty-FP protections fail against both controls. Low-count F1 also fails against F; the small H–U decline is within its tolerance. Both group protections fail: March 12 H MAE 5416.6 versus F 78.9333 and U 1016.9333; February 27 H MAE 1788.125 versus F 107.375 and U 621.5. Secondary relative and absolute gates fail.

## Fitting behavior and interpretation

| Arm | Fitting MAE at 0.50 | Fitting positive-frame F1@8 | TP / FP / FN, 338 truth points |
|---|---:|---:|---:|
| F | 8.0238 | 0.0256 | 1 / 0 / 337 |
| U | 362.5238 | 0.0350 | 337 / 15227 / 1 |
| H | 8.0476 | 0.0000 | 0 / 0 / 338 |

The continuation did not learn useful confident point discrimination even on the 42 fitting frames. U demonstrates that raising scores is possible, but it raises scores for many incorrect or duplicate proposals. H suppresses primary detections and leaves many subthreshold proposals. The evidence rejects this particular loss/sampling repair at the frozen budget; it does not establish a unique causal explanation, prove that no different architecture can learn, or assess convergence from fresh initialization. It also does not justify selecting an alternative threshold from these development results.

This is one matched continuation with repeated draws and one seed. A common restarted learning-rate schedule and identical authentic Adam history controlled additional optimization across arms. Architecture, labels, dynamic assignment, point localization loss, augmentation, public replay and thresholds were unchanged. The five reserved recording groups were untouched. Generic BCE or hard-negative mining is existing methodology, not a novelty result.

## Verification and decision

Independently checked all **134 frozen files**, the original P11 resume/counter hashes and CUDA guard receipts, completed controller and three live Docker exits, all 200 sequential optimizer records and 1600 identical augmentation records per arm, native truth and unique prediction coverage for every fitting/development image, result/checkpoint hashes and literal cumulative/new step metadata. Recomputed count errors, MAE/RMSE/bias, GAME1/2 and maximum point matching from saved arrays. A separate count-arithmetic pass agreed. Tensor finiteness was checked by the frozen worker; the independent verifier did not deserialize model tensors. The original analyzer/collector and helper hashes matched the launch receipt.

**Decision: stop this recipe here. No full training, additional continuation, threshold search, new seed or new experiment was launched.** Preserve the failed results and controls. The completion monitor is closed after reporting this verified result.

Result receipt · Full independent analysis · Supplemental verification · Original checkpoint identities

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 47 files under the folder ipic-objectness/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
