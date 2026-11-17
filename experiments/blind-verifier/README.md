# Held-out visible-sack counting and localization

**On 51 held-out frames containing 644 reviewed sacks, the frozen supplemental verifier reduced count MAE from 6.12 to 5.06 sacks per frame (17.3%). MAE improved in all five recording groups.** At the same number of predictions per frame, the verifier matched 167 sacks and raw-score ranking matched 165.

## Cohort and scoring

The system was selected on development data. Its weights, operating point, voting procedure and equal-budget control were frozen before held-out inference. The DGX first reproduced the development predictions, then sealed the three sets of held-out predictions. The blind review export covered every frame. Before the archive was opened, the owner resolved H025's conflicting five marks and empty-frame note in favor of zero sacks. The other 50 labels were preserved as exported. The original and adjudicated exports are retained. No model was retrained or threshold changed for scoring.

The target is a distinguishable filled sack on the selected truck or carried by its associated workers. A correct localization is a one-to-one point match within eight pixels after scaling the image long side to 1,024 pixels. All reviewed centers were included, including one each in H001 and H077 outside the model's fixed inference region. GAME1 and GAME2 measure spatial count error in 2×2 and 4×4 subdivisions with the frozen edge convention. Label adjudication · Frozen protocol

| Method | Predictions | Matches | False positives | Misses | F1@8 | MAE ↓ | RMSE ↓ | Bias | GAME1 ↓ | GAME2 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Voting baseline | 338 | 150 | 188 | 494 | .3055 | 6.12 | 7.72 | −6.00 | 6.67 | 7.33 |
| Frozen verifier | 408 | 167 | 241 | 477 | .3175 | 5.06 | 6.53 | −4.63 | 5.80 | 6.67 |
| Equal-budget raw ranking | 408 | 165 | 243 | 479 | .3137 | 5.06 | 6.53 | −4.63 | 5.80 | 6.78 |

The verifier added 70 predictions over voting, producing 17 additional matches and 53 additional false positives. Recall rose from 23.3% to 25.9%, while precision fell from 44.4% to 40.9%. The raw-ranking control accepted exactly the verifier's per-frame number of additions. Its count errors therefore coincide with the verifier's. The verifier gained one match on three frames, lost one match on one frame, and tied on the other 47. The equal-budget F1 difference was +0.0038.

## Low counts and empty frames

The 29 low-count frames (1–11 sacks) contained 152 reviewed sacks. Voting missed 126 and entirely missed 13 frames. The verifier and equal-budget control each missed 121 and entirely missed 12 frames. Low-count MAE improved from 3.66 to 3.14 and pooled F1 from .263 to .291 versus voting. All three methods predicted zero on H025, the only reviewed empty frame.

## Results by recording group

| Archive group | Frames | Reviewed sacks | Voting MAE | Verifier and equal-budget MAE | Voting matches | Verifier matches | Equal-budget matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| February 28 | 2 | 27 | 8.50 | 7.50 | 6 | 6 | 7 |
| March 3 | 11 | 76 | 2.73 | 2.36 | 24 | 27 | 27 |
| March 5 | 19 | 176 | 5.79 | 4.58 | 25 | 27 | 27 |
| March 6 | 4 | 49 | 3.75 | 2.75 | 13 | 13 | 13 |
| March 7–8 archive group | 15 | 316 | 9.33 | 7.93 | 82 | 94 | 91 |

## What the paper can claim

On the frozen benchmark split, supplementing the voting baseline improved visible-sack count accuracy across all five held-out groups. Most of the additional localized sacks were also recovered by raw ranking when given the verifier's per-frame budget. The verifier's placement advantage was two matches overall, with a gain in one recording group, a loss in another and ties in three. The evidence for counting improvement is stronger than the evidence for a better ranking method. The cohort contains related frames and only one empty example. Independent-site performance, reliable rejection of empty trucks and physical inventory behind occlusions remain unmeasured.

The complete reviewed cohort comprises 116 frames across ten dependency groups, with 42 fitting, 23 development and 51 held out. The holdout results above belong to the verifier that was selected on development data. The separate paired-context architecture has its own development results. On November 10 the project owner confirmed permission to release the benchmark dataset. Permission record

Per-frame results and recording-level metrics · Independent verification of all 153 method/frame scores

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.


[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
