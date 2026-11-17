# CounTR sack pilot: verified result — November 5, 2026

**Both arms completed; relative and absolute gates FAILED. No full training follows.**

Verified at 2026-11-05T13:24:43.635595+09:00. Each trained arm saved all 300 updates (600 total), all 1,200 input records matched across arms, both containers/controller exited 0 without OOM or a guard stop. Execution finished approximately 12:49:55 KST. All 86 frozen code/data files, official pretrained weights, final checkpoint hashes/literal metadata and all 195 density arrays were verified. Independent count/GAME1/2/3 arithmetic on all arrays and fine spatial spot-checks agreed within 5.1e-13. Tensor finiteness is a frozen-worker check; local analysis does not deserialize model tensors.

The endpoint cohort is 23 reviewed development frames / 355 points from two recording groups:18 positive, 5 empty;3 low-count(1–11) and15 higher-count(12–36). The same 42 fitting images / 338 points across 3 groups were used in both arms. Five reserved groups were not used.

| Metric | Frozen pretrained B | Reset decoder R | Fine-tuned pretrained P |
|---|---:|---:|---:|
| MAE ↓ | 9.0393 | 10.0087 | 7.6703 |
| RMSE ↓ | 12.5801 | 12.3124 | 10.2154 |
| Signed bias | 0.9357 | 6.4284 | 3.2381 |
| Higher-count MAE ↓ | 11.9402 | 8.8808 | 7.2589 |
| Low-count MAE ↓ | 2.4885 | 19.5293 | 12.8044 |
| Mean count on empty images ↓ | 4.2671 | 7.6801 | 5.8242 |
| GAME1 ↓ | 11.8858 | 11.0756 | 10.1703 |
| GAME2 ↓ | 14.2161 | 16.4848 | 13.7056 |
| GAME3 ↓ | 15.9639 | 19.8979 | 15.9580 |
| Spatial mass precision@8 ↑ | 12.11% | 4.32% | 6.81% |
| Spatial mass recall@8 ↑ | 13.76% | 5.35% | 7.48% |

Spatial mass precision/recall are positive-image macro averages within 8 pixels in a long-side1024 chart. Each density pixel can be credited to only one nearest reviewed point. They are **not detection F1 or discrete instance precision/recall**. GAME1/2/3 compare spatial count bins at 2×2, 4×4, 8×8; lower is better.

## What improved and what failed

P lowers overall MAE by 1.3690 (15.15%) against frozen B and 2.3384 (23.36%) against reset R. Higher-count MAE falls 39.21% / 18.26% respectively. RMSE and aggregate GAME1/2/3 also improve; the GAME3 change versus B is only−0.0059 and is practically negligible. Both recording-group MAEs pass protection. P passes every relative comparison against R. This supports a benefit from counting-decoder pretraining at this fixed budget.

P fails protection against B on **low-count MAE and GAME1/2/3, spatial mass precision and recall, and empty-image mass**. Low-count MAE rises 2.4885 → 12.8044 (+10.3159). Low-count GAME1/2/3 change by +7.4492 / +7.5810 / +8.7061. Empty-image mean count rises 4.2671 → 5.8242 (+1.5570); total empty mass rises 21.3357 → 29.1209 (+7.7852), exceeding the 0.1 allowance. Spatial mass precision loses 5.29 percentage points and recall 6.28 points, beyond the 1-point tolerance.

The three low-count truths 2, 2, 7 become P estimates 15.8622, 14.9946, 18.5563. The empty dev_00 frame receives 16.2497 expected sacks. A near-correct count also need not mean correct positions: dev_01 has truth 21 and prediction 21.0797, but spatial mass precision/recall are only 8.49% / 8.52%, and GAME3 is 17.8613.

P meets the absolute MAE threshold 7.7174 with 7.6703, but fails all other absolute checks: empty mean 5.8242 > 0.5; fitting MAE 3.4456 > 2; spatial precision 0.0681 < 0.5 and recall 0.0748 < 0.5. Fitting MAE is 7.7877 / 4.9898 / 3.4456 for B/R/P. Thus **relative FAIL, absolute FAIL, combined FAIL**.

## Recording groups and raw-density disclosure

| Group | B MAE | R MAE | P MAE |
|---|---:|---:|---:|
| March 12, 15 images | 8.2411 | 11.5479 | 8.0068 |
| February 27, 8 images | 10.5360 | 7.1227 | 7.0394 |

All primary counts use the prespecified nonnegative density integration max(raw,0)/60. Total predicted development mass is 376.5207 / 502.8541 / 429.4761 for B/R/P, versus 355 truth. Signed raw totals are 373.7827 / 407.8763 / 359.6433; negative mass removed is 2.7380 / 94.9779 / 69.8328. These raw diagnostics are preserved, not used to select an alternative scoring rule.

The new outputs are nonzero and aggregate count error is lower, but the density is insufficiently concentrated at the reviewed sack locations and overcounts sparse/empty scenes. That is evidence of a count/localization tradeoff under this recipe, not proof of a unique training cause. No threshold, learning-rate extension, new seed or further model was selected.

## Decision and limits

**Stop this bounded recipe here and retain all results. Full training is not justified by the frozen gate.** CounTR and decoder fine-tuning are established methods; this is not novelty or paper readiness. The small reused development groups, one seed and limited budget do not establish independent generalization, convergence, FSC dense/extreme performance or failure of all density methods.

Frozen plan · Result receipt · Independent analysis · Additional arithmetic checks

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 212 files under the folder countr-pretrained-decoder/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
