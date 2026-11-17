# Does carried-sack detection and whole-image count supervision improve foreground sack counting?

**Completed November 10. The joint model improved count MAE on the previously scored 51-frame test split, with more matches and lower localization F1 than the frozen verifier.** Test MAE fell from 5.06 to 4.22 and matches increased from 167 to 175. The model added 81 points for eight additional matches. The matched benchmark-only head achieved MAE 4.00 and 173 matches. The joint model therefore falls short of a combined counting and localization improvement attributable to auxiliary supervision.

## Dataset and split

| Component | Training data used in this study | Supervised output |
|---|---|---|
| Roboflow Sackdetection v1, native training split | 21,006 exported images, 14,403 boxes, 7,002 source filename stems | Presence, center and dimensions of a carried or handled sack |
| Reviewed whole-image inventory | All 300 approved totals, summing to 10,462 visible sacks across frames | Total count across the entire image |
| Recording-grouped visible-sack benchmark | 42 fitting frames from 3 groups, 338 reviewed centers | Foreground-truck center positions, regional counts and total count |
| Benchmark evaluation | 23 development frames from 2 groups and 51 previously scored test frames from 5 groups | Count error, one-to-one point localization and spatial count error |

The original Roboflow export contains 24,007 images. Its training split contains three versions per source filename stem. Of the 300 reviewed count images, 150 belong to the native training split and 150 to native validation. All 300 totals are explicit fitting supervision. Native Roboflow test images are excluded from model fitting and scoring in this study. Across the 300 reviewed images, 111 with no carried-sack box contain visible sacks. Their approved positive inventory totals are used in training. Among the 21,006 carried-task images, 6,723 supply a negative carried-objectness target. A missing inventory count stays unknown.

## Matched training experiment

All new trainable weights see sack imagery exclusively. The visual encoder uses fixed official DINOv2 ViT-B/14 weights with registers. The original verifier is the frozen production comparison from the November 9 experiment. Its components also have public pretraining. The foreground branch receives the existing benchmark ROI. Both external auxiliary tasks receive the entire Roboflow image. Inputs to the new heads are resized to 336×336.

The four conditions share their initialization, visual features, frame order, architecture, AdamW schedule and optimization budget. Every arm learns from the 42 benchmark fitting frames. The first arm uses those point labels alone. The next two arms add the carried-box loss or the reviewed-count loss individually. The prespecified primary arm adds both. The box loss supervises carried-sack objectness, subcell offsets and box size. It supplies no all-visible background or truck-count target. The inventory branch predicts a center score, subcell offset and positive spatial mass. Reviewed whole-image totals supervise only the integrated mass; they are never converted into locations. Benchmark point labels provide both exact positions and local mass targets.

Each arm receives 657 updates. Each step processes up to 32 box-task frames, two whole-image count examples and two foreground fitting frames. The native Roboflow training images are traversed exactly once. Count and point examples cycle through shuffled orders for 1,314 presentations of each supervision type per arm. Auxiliary loss coefficients are 0.5 when enabled and zero otherwise. The seed is 1729.

## Endpoint and comparison

The frozen verifier's output is the starting set for all four conditions. The candidate head can add up to 64 centers when its rounded inventory estimate exceeds the number of existing points. Additions must be separated from existing centers by more than eight pixels at a standardized long side of 1,024 pixels. Each arm's minimum mass for admitting additions is set using only the three reviewed empty fitting frames. Its parameters and operating point are saved before the new development or test predictions are generated.

The additional equal-budget control uses the point-only head's candidate ordering and locations at the joint arm's exact per-frame quota. The comparison with the frozen verifier measures total system change. The comparison with the point-only control measures the incremental effect of auxiliary supervision at matched training compute. The equal-budget comparison tests placement with the allocated count fixed.

All three task definitions concern sacks. The box export and the foreground point benchmark have different annotation scopes and split contracts. Adjacent Roboflow frames cross the original training and validation splits. The 51 benchmark test frames were used in an earlier reported comparison. Their labels stay out of the new models' fitting and cutoff selection, but these follow-up results cannot serve as an independent confirmation on newly untouched recordings.

## Benchmark results

All four arms completed 657 updates for 2,628 updates in total. The training ledger accounted for every native training image exactly once per arm, all 300 reviewed totals, and all 42 foreground fitting frames. The worker checked finite gradients, model parameters and optimizer state. The four checkpoints and operating points were frozen before scoring. The guarded DGX container exited successfully after about 11 minutes 15 seconds. A second maximum-matching implementation agreed on the number of correct detections for every evaluated frame and method.

Each row with added supervision starts with the same frozen verifier. “Points” includes the 42 benchmark fitting frames and their derived regional/global counts. The boxes-only and totals-only additions are ablation conditions; the planned primary condition uses both.

| Test condition (51 frames, 644 sacks) | Predictions | Matches | False positives | F1@8 ↑ | MAE ↓ | RMSE ↓ | GAME1 ↓ | GAME2 ↓ | Low-count misses / 152 | False points on the one empty frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen verifier | 408 | 167 | 241 | .3175 | 5.06 | 6.53 | 5.80 | 6.67 | 121 | 0 |
| + points only | 488 | 173 | 315 | .3057 | 4.00 | 5.39 | 5.45 | 6.59 | 116 | 1 |
| + points and carried boxes | 473 | 174 | 299 | .3115 | 3.98 | 5.56 | 5.31 | 6.61 | 114 | 1 |
| + points and reviewed totals | 667 | 196 | 471 | .2990 | 3.31 | 4.40 | 5.24 | 7.16 | 114 | 4 |
| **+ points, boxes and totals** | **489** | **175** | **314** | **.3089** | **4.22** | **5.66** | **5.55** | **6.73** | **114** | **2** |
| Points-only rank at joint per-frame count | 489 | 172 | 317 | .3036 | 4.22 | 5.66 | 5.63 | 6.73 | 118 | 2 |

On the test split, the joint condition reduced MAE by 0.84 sacks per frame (16.7%) versus the verifier. It lowered MAE in four of five groups and tied in one. Point F1 decreased in every group. At exactly 489 predictions distributed identically across frames, the joint head localized three more sacks than the points-only rank. On the 29 low-count frames it matched 38 sacks versus the verifier's 31, reduced MAE from 3.14 to 2.31, and entirely missed 10 positive frames versus 12. The joint condition produced two false points on the only empty test frame.

| Test recording group | Frames | Verifier MAE | Joint MAE | Verifier matches | Joint matches | Points-only matches at joint count |
|---|---:|---:|---:|---:|---:|---:|
| February 28 | 2 | 7.50 | 6.50 | 6 | 6 | 6 |
| March 3 | 11 | 2.36 | 2.36 | 27 | 29 | 27 |
| March 5 | 19 | 4.58 | 3.32 | 27 | 31 | 30 |
| March 6 | 4 | 2.75 | 1.50 | 13 | 13 | 13 |
| March 7–8 archive group | 15 | 7.93 | 7.13 | 94 | 96 | 96 |

| Development condition (23 frames, 355 sacks) | MAE ↓ | Matches | F1@8 ↑ | Low-count misses / 11 | False points on 5 empty frames |
|---|---:|---:|---:|---:|---:|
| Frozen verifier | 6.35 | 88 | .2933 | 11 | 0 |
| + points only | 5.83 | 90 | .2932 | 11 | 1 |
| + points and carried boxes | 6.00 | 88 | .2876 | 11 | 2 |
| + points and reviewed totals | 6.17 | 105 | .2593 | 10 | 5 |
| **+ points, boxes and totals** | **6.57** | **104** | **.2719** | **9** | **6** |
| Points-only rank at joint per-frame count | 6.57 | 108 | .2824 | 10 | 6 |

## Paper interpretation

This is a sack-only benchmark study of how annotation scope affects transfer. The primary joint condition recovered additional low-count sacks. Its aggregate count gain over the verifier on the reused test set did not repeat on development, and the matched benchmark-only head achieved lower MAE on both splits. The counts-only auxiliary condition produced the lowest test MAE and the most false detections. These data support reporting counting and localization separately. They do not support replacing the verified benchmark reference with the joint head or promoting the counts-only condition after observing its test result. Empty-truck rejection also remains an open issue.

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 7 files under the folder auxiliary-supervision/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
