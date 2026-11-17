# FIRC ordinary sack training and development results

**Training completed. The final checkpoint failed the practical readiness criteria.** At the threshold chosen from training frames, development MAE was 11.83 sacks per frame. The model localized 25 of 355 reviewed sacks within eight pixels in the long-side-1024 coordinate system.

The run saved all 800 public foundation updates and all 3,000 adaptation updates. The 30,400 image presentations matched the training plan. The controller and training container exited successfully after approximately 6 hours 44 minutes on November 6. The final checkpoint passed all 29 training verification checks.

Development evaluation was still pending when checked on November 8. The first local launcher exhausted PC memory while building its upload. A streamed transfer verified the frozen release hashes on DGX. One evaluation processed 42 training and 23 development frames, passed all 20 verification checks and made zero optimizer updates. The training frames selected 0.35 by minimum count MAE. The saved threshold receipt preceded all 23 development predictions.

| Development metric | Fixed 0.50 | Training-selected 0.35 | Change |
|---|---:|---:|---:|
| MAE, sacks/frame | 15.0870 | 11.8261 | −3.2609 |
| RMSE | 19.0685 | 15.2401 | −3.8284 |
| Bias, predicted minus reviewed | −15.0870 | −11.8261 | +3.2609 |
| Pooled point F1@8 | 0.0275 | 0.1142 | +0.0866 |
| Positive-image macro F1@8 | 0.0241 | 0.0887 | +0.0646 |
| GAME1 | 15.0870 | 11.8261 | −3.2609 |
| GAME2 | 15.1739 | 12.3478 | −2.8261 |
| True positives / false positives / missed sacks | 5 / 3 / 350 | 25 / 58 / 330 | +20 / +55 / −20 |
| Positive frames with a match, out of 18 | 5 | 10 | +5 |

F1 uses one-to-one matching between predicted points and reviewed points. GAME1 and GAME2 sum absolute count errors across 2×2 and 4×4 grids in the foreground region. The calibrated model predicted 83 sacks across the 23 frames, giving a net undercount of 272.

| Subset / metric | Fixed 0.50 | Training-selected 0.35 | Change |
|---|---:|---:|---:|
| Three low-count frames, MAE | 3.6667 | 3.6667 | 0 |
| Low-count positive macro F1@8 | 0 | 0 | 0 |
| Low-count GAME1 / GAME2 | 3.6667 / 3.6667 | 3.6667 / 3.6667 | 0 / 0 |
| Five empty frames, false positives | 0 | 0 | 0 |
| Fifteen frames with 12–36 sacks, MAE | 22.4000 | 17.4000 | −5.0000 |
| Higher-count GAME1 / GAME2 | 22.4000 / 22.5333 | 17.4000 / 18.2000 | −5.0000 / −4.3333 |
| Recording group 483475312362, MAE | 17.6667 | 13.8667 | −3.8000 |
| Recording group b8c3dda6893f, MAE | 10.2500 | 8.0000 | −2.2500 |

The calibrated model missed all 11 sacks in the three low-count frames. It met the zero-false-positive requirement on empty images. Positive-image macro F1 was 0.9533 on the 42 training frames and 0.0887 on development. Training MAE was 0.9286.

| Frozen readiness criterion | Observed | Outcome |
|---|---:|---|
| Development MAE ≤7.7174 | 11.8261 | Fail |
| Pooled F1@8 ≥0.50 | 0.1142 | Fail |
| Positive-image macro F1@8 ≥0.50 | 0.0887 | Fail |
| At least 14 positive frames with a match | 10/18 | Fail |
| Training positive-image macro F1@8 ≥0.80 | 0.9533 | Pass |
| Low-count MAE ≤2 | 3.6667 | Fail |
| Low-count positive macro F1@8 ≥0.50 | 0 | Fail |
| Empty-frame false positives = 0 | 0 | Pass |

The earlier four-image reference achieved calibrated development MAE 14.0870 and one match on this cohort. The broader training run reduced MAE by 2.2609 and added 24 matches in a descriptive comparison across different training recipes. The earlier adapted CounTR model recorded MAE 7.6703 on these same 23 frames under a different training budget. Its density-based spatial scores measure a different quantity from point F1.

These results cover ordinary visible-sack supervision, one seed and two reused development recording groups. The five reserved groups remain unscored and independent deployment accuracy remains unmeasured.

Full metrics and per-image detections · Dated result and file identities · Training verification

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 71 files under the folder firc-endpoint/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
