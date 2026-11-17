# FIRC training-only threshold calibration: verified result — November 6, 2026

**Primary relative gate: FAIL. Absolute readiness gate: FAIL. Full training has not started.**

Verified 2026-11-06T14:59:09.848037+09:00. Both model checkpoints completed all 130 prescribed query-image forwards and eight separately logged reference-image encodings, with zero optimizer/persistent steps. Controller and Docker completed with exit 0, no OOM and no guard stop. All 31 independent receipt, input, coverage, parity and selection checks passed. Worker finite-state and unchanged-parameter checks passed; the local verifier does not independently deserialize checkpoint tensors.

Thresholds were chosen using all 42 training-frame counts only, then saved after exactly 84 training forwards and before the first development forward. The worker did not receive development point/count labels. The 23 development frames (355 reviewed sacks; two recording groups) were scored only after that freeze.

| Model / inference | Global threshold | Dev MAE | RMSE | Pooled F1@8 | Positive macro F1@8 | GAME1 | GAME2 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| reference_lr / fixed | 0.50 | 14.7826 | 18.4509 | 0.005405 | 0.002469 | 14.7826 | 14.7826 | 1 / 14 / 354 |
| reference_lr / calibrated | 0.35 | 14.0870 | 17.6758 | 0.005181 | 0.002268 | 14.4348 | 14.6957 | 1 / 30 / 354 |
| lower_lr / fixed | 0.50 | 14.2174 | 17.9480 | 0.005222 | 0.002179 | 15.0000 | 15.2609 | 1 / 27 / 354 |
| lower_lr / calibrated | 0.85 | 15.4348 | 19.4254 | 0.000000 | 0.000000 | 15.4348 | 15.4348 | 0 / 0 / 355 |

F1@8 is maximum one-to-one point matching in a long-side-1024 native-image chart. GAME1/2 are spatial count errors in the predeclared coarse foreground ROI. Empty-empty frames receive no positive-image F1 credit.

## Signed effects and protections

| Calibrated minus fixed 0.50 | Dev MAE | Dev RMSE | Dev GAME1 | Dev GAME2 | Positive macro F1 |
|---|---:|---:|---:|---:|---:|
| reference_lr | -0.6957 | -0.7751 | -0.3478 | -0.0870 | -0.000202 |
| lower_lr | +1.2174 | +1.4775 | +0.4348 | +0.1739 | -0.002179 |

The reference MAE reduction is 0.6957 (4.71%), below the frozen 10% requirement of 1.4783. Its 16 additional development detections are all unmatched at the specified eight-pixel tolerance: TP stays at one, FP rises from 14 to 30, and FN stays at 354. Lower count error here does not establish improved localization. GAME improves slightly in aggregate, but pooled and positive-image F1 decline.

All four combinations predict zero on the three low-count frames (11 sacks total): MAE 3.6667, GAME1/2 3.6667, F1 0. All also predict zero on the five empty development frames. Low-count and empty-image nonregression therefore passes without demonstrating useful low-count performance.

| Model / inference | Higher-count MAE (12–36) | Higher GAME1 / GAME2 | Positive development frames with TP |
|---|---:|---|---:|
| reference_lr / fixed | 21.9333 | 21.9333 / 21.9333 | 1/18 |
| reference_lr / calibrated | 20.8667 | 21.4000 / 21.8000 | 1/18 |
| lower_lr / fixed | 21.0667 | 22.2667 / 22.6667 | 1/18 |
| lower_lr / calibrated | 22.9333 | 22.9333 / 22.9333 | 0/18 |

These sacks are not an FSC dense/extreme-count test; the development maximum is 36.

| Development recording group | Reference fixed → calibrated MAE | Lower fixed → calibrated MAE |
|---|---:|---:|
| ipic_dependency_483475312362 | 17.6000 → 16.9333 | 17.1333 → 17.9333 |
| ipic_dependency_b8c3dda6893f | 9.5000 → 8.7500 | 8.7500 → 10.7500 |

## Every failed component

- reference_lr, relative: mae_gain, positive_macro_f1_gain.
- reference_lr, absolute: mae, pooled_f1, positive_macro_f1, positive_frames_tp, fitting_positive_macro_f1.
- lower_lr, relative: mae_gain, positive_macro_f1_gain, development_rmse, development_game1, development_game2, group_ipic_dependency_483475312362, group_ipic_dependency_b8c3dda6893f.
- lower_lr, absolute: mae, pooled_f1, positive_macro_f1, positive_frames_tp, fitting_positive_macro_f1.

Absolute requirements remain MAE ≤7.7174, pooled and positive macro F1 ≥0.50, TP on at least 14/18 positive development frames, training positive macro F1 ≥0.80, and zero empty false positives. The reference protection components pass, but the required gains and useful absolute performance do not. The lower-LR calibrated model predicts zero on every development image; it also worsens RMSE, GAME1/2 and both group MAEs beyond tolerance. It is not promoted as an alternative.

## What the failure actually establishes

**Distinguish optimization exposure from calibration exposure.** These heads were optimized on only four images, not on all 42 training frames. All 42 frames were used for threshold selection; the other 38 supplied no optimizer updates. This result rejects a calibration-only deployment shortcut. It does not prove that properly training on the broader cohort cannot work.

| Calibrated model / optimization exposure | MAE | Pooled F1@8 | Positive macro F1@8 | TP / FP / FN |
|---|---:|---:|---:|---|
| reference_lr / four_optimized_images | 0.0000 | 1.000000 | 1.000000 | 56 / 0 / 0 |
| reference_lr / other_38_training_images | 6.9474 | 0.018987 | 0.005235 | 3 / 31 / 279 |
| lower_lr / four_optimized_images | 14.0000 | 0.000000 | 0.000000 | 0 / 0 / 56 |
| lower_lr / other_38_training_images | 7.2632 | 0.006897 | 0.002058 | 1 / 7 / 281 |

The reference head counts and localizes all 56 targets on its four optimized images at the training-selected 0.35 threshold. Its failure on other training images and development images shows that the current useful ranking is highly specific to the repeatedly fitted inputs. A global threshold does not repair that lack of transfer. The lower-LR training-MAE selector chooses 0.85, suppressing its false detections but collapsing development predictions to zero. We preserve that outcome rather than changing the selection objective after seeing development.

## Training and paper decision

Do not promote either diagnostic checkpoint to research initialization or start the full four-state experiment from this failed calibration gate. The owner requested actual training; the readiness condition is not satisfied, so no longer optimization was launched. No old experiment was restarted.

The next necessary training question is broader ordinary-image learning with a fresh counting head and the official pretrained encoder, using reviewed training data and valid training-only reference semantics. It is not another threshold sweep, LR replay or four-image continuation. The existing public data capacity audit can be reused, but a production runner still needs correct target-specific D2S references, matched augmentation geometry and mixed-data CUDA recovery before an ordinary public foundation run. Those are readiness tasks, not completed training. The proposed physical four-state method additionally has zero admitted quartets; ordinary-image success cannot substitute for that mechanism test.

Final deployable accuracy remains unmeasured. Five reserved recording groups remain unscored, and their 51 prepared frames have zero approved exhaustive labels. A frozen model, threshold, inference inputs and reviewed final labels are needed before that one-time evaluation. The two reused development groups and this standard calibration check do not establish novelty, independent generalization, paper acceptance or hidden-inventory accuracy.

## Evidence

- Frozen plan SHA256: `a7789823e84283634b00919dfcd48a2815f44580bb595bfcdf9432674dd919e7`.
- Output archive SHA256: `8225837f3f0bedf6cd4808cb8f69c0845c5d5199ff4bd82517ff09933792f832`.
- Independent analysis SHA256: `56587074a2c1f4e9d26fe5dd048fdc9ee5e4713d3d7981f5e9d225ecaf304f38`.
- Threshold receipt SHA256: `bf3c0541d737cc76b5dcb2c83d05a620d193dca3c12349fd41844c31936a5c41`.
- Full per-image predictions, all 17 threshold candidates per model, all checks, gates and signed deltas are preserved in `artifacts/firc_calibration_v1_20261106/`.
- The prior learning-rate experiment and its original fixed-0.50 failures remain unchanged.

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 136 files under the folder firc-calibration/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
