# Count error and sack localization with a pretrained detector

**November 10, 2026. The primary query adapter failed the overall comparison.** The unadapted CountGD++ detector localized 351 of 644 reviewed sacks on the 51 previously scored test frames, including 77 of the 152 low-count sacks. It made 1,487 predictions. The primary trained head returned zero on all 51 frames at its frozen threshold.

## Matched detector experiment

Every detector row used the published CountGD++ weights, the same foreground crop, the word “sack” and the first fixed training exemplar in the released benchmark (frame 004 with its original box and surrounding training-image context). This was global inference. The native comparator used the author’s 0.23 operating point. For every frame the worker reproduced the author’s filtered box list from the captured raw outputs. External visual prompts are an established capability of this model. [CountGD++ paper](https://arxiv.org/abs/2512.23351) · [Author implementation](https://github.com/niki-amini-naieni/CountGDPlusPlus)

Each adapted head received all 900 candidate boxes before filtering. Both learned center corrections and confidence scores from the same reviewed sack points, under the same one-to-one matching rule, training draws, initialization, loss and optimizer schedule. The control used box geometry and token confidence. The primary also used the 256-dimensional detector embedding for each candidate. The frozen detector had no weight updates. The 42 fitting frames supplied 338 centers across three recording groups. Three recording-excluded head fits determined one threshold per model. A final fit used all 42 frames. The detector prompt was fixed across all fits, including the source recording of that prompt.

DGX completed all 116 model forwards, 250 optimizer updates per head per fit and 2,000 updates in total. The container exited successfully. The local verifier checked the input and result receipts, fold memberships, updates and checkpoint hashes. Two point-matching implementations agreed on every scored frame and method. The threshold and head weights were saved before scoring the 23 development and 51 test frames. These datasets had already informed research decisions. Frozen design · Execution receipt

## Results

| Split / system | MAE ↓ | Pooled F1@8 ↑ | Matches | Unmatched predictions | GAME1 / GAME2 ↓ | Low-count matches | Predictions on empty frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development · frozen verifier | 6.35 | .2933 | 88 | 157 | 7.74 / 9.57 | 0 | 0 |
| Development · previous refinement | 5.61 | .3126 | 98 | 174 | 7.52 / 9.35 | 0 | 0 |
| Development · CountGD++ | 33.22 | .3057 | 221 | 870 | 33.91 / 35.13 | 8 | 8 |
| Development · geometry and scores | 12.57 | .2248 | 68 | 182 | 13.52 / 14.65 | 0 | 0 |
| Development · primary query adapter | 15.13 | .0110 | 2 | 5 | 15.13 / 15.13 | 0 | 0 |
| Reused test · frozen verifier | 5.06 | .3175 | 167 | 241 | 5.80 / 6.67 | 31 | 0 |
| Reused test · previous refinement | 4.67 | .3210 | 173 | 261 | 5.65 / 6.63 | 36 | 2 |
| Reused test · CountGD++ | 16.61 | .3294 | 351 | 1,136 | 17.35 / 18.22 | 77 | 3 |
| Reused test · geometry and scores | 9.92 | .2168 | 85 | 55 | 10.00 / 10.08 | 0 | 0 |
| Reused test · primary query adapter | 12.63 | .0000 | 0 | 0 | 12.63 / 12.63 | 0 | 0 |

Development contains 355 reviewed sacks, including 11 on low-count frames, and five empty frames. The test contains 644 reviewed sacks, including 152 on low-count frames, and one empty frame. A detection must match a unique reviewed center within eight pixels at a full-image long side of 1,024. An unmatched prediction can reflect a duplicate, displacement or a background detection.

The primary threshold of 0.9394 was set just above the maximum score on the three empty fitting frames under recording-excluded fits. This was the lowest permitted threshold. Seven proposals survived on development and zero survived on test. Among the seven development predictions, the original coordinates matched the same two sacks as the corrected coordinates. Native ranking at the same per-frame counts matched four. The prespecified count, localization and spatial accuracy criteria failed. The score threshold protected the empties while suppressing positives on new recordings. The experiment does not identify a unique underlying cause of the score shift. Per-frame comparisons and decision checks

The previous boxes-and-totals study supplied a different tradeoff. The auxiliary condition that used reviewed totals alongside fitting points attained test MAE 3.31, F1 .2990 and four false detections on the one empty test frame. Its MAE and the native detector’s F1 belong to different systems. A separate earlier two-exemplar, full-image protocol gave development MAE 11.74 and pooled F1 .1447 for CountGD++, and 13.57 and .0686 for the GeCo2 external-reference adapter. The different input and prompting protocols should appear explicitly in any cross-system table. Auxiliary study · Earlier two-exemplar results

## Implications for a benchmark paper

At the original per-frame budgets totaling 408 predictions, perfect label-informed selection among CountGD++ centers could match 297 sacks. That calculation uses different candidates from the frozen verifier. Subtracting the verifier's 167 matches from 297 combines proposal generation and ranking effects. On the verifier's own eligible candidates, with its voting seeds retained and the same per-frame budgets, the oracle matched 212. Raw ranking matched 165 and learned ranking matched 167. The measured gap to this oracle is 45 matches for the learned verifier. The oracle uses ground truth. CountGD++ ceiling · Same-candidate replay

The original blind result remains the strongest evidence for a counting improvement over voting. The follow-up models quantify the difficulty of obtaining accurate counts and individually correct centers from a single system. A claim that one model leads overall is unsupported. Original blind comparison

The figure shows the sole empty test frame and a positive test frame chosen by proximity to the median positive count with ties broken by image hash. Predictions did not enter this choice. Frame-level comparison

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 11 files under the folder countgd-query/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
