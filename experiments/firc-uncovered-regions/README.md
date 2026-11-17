# Moving predictions to uncovered sack regions at a fixed count

**The proposed reallocation produced 61 matches, down from both the 71-match confidence ranking and the 75-match center refinement.** It lost 15 matches on March 12 relative to the 75-match starting positions and gained one on February 27. The February result also fell one match below the original confidence ranking. Neither recording beat both references.

| Method | Predictions moved | March 12 matches (269 sacks) | February 27 matches (86 sacks) | Total matches (355 sacks) | Pooled F1@8 | GAME1 / GAME2 |
|---|---:|---:|---:|---:|---:|---:|
| Saved confidence ranking | — | 62 | 9 | 71 | 0.250 | 8.696 / 10.174 |
| Saved center refinement, starting positions | — | 68 | 7 | 75 | 0.264 | 8.696 / 10.087 |
| **Reallocate using FIRC and CounTR coverage** | **126** | **53** | **8** | **61** | **0.215** | **8.870 / 10.174** |
| Coverage using FIRC alone | 73 | 73 | 5 | 78 | 0.275 | 8.609 / 9.652 |
| Primary removals, confidence-ranked destinations | 126 | 69 | 7 | 76 | 0.268 | 8.261 / 9.565 |
| Primary destinations, remove lowest-confidence seeds | 126 | 49 | 9 | 58 | 0.204 | 8.522 / 9.565 |
| Rank both removals and destinations by confidence | 126 | 65 | 8 | 73 | 0.257 | 8.261 / 9.478 |

## What moved

Every arm used the same number of predictions on each frame, with 213 predictions across the 23 development frames. The primary rule started from the saved 75-match coordinates. It considered FIRC candidates with confidence at least 0.025 and measured local evidence from the FIRC confidence and CounTR-P density. Each selected point covered evidence within eight pixels on the long-side-1024 chart. An overlapping point had little unique coverage. An isolated point could also have little evidence around it.

The rule exchanged the least costly point to remove for the eligible destination with the largest uncovered evidence mass, while the exchange increased total coverage. Each destination was more than 16 pixels from every starting point and every other destination. Retained points kept their exact coordinates. “Uncovered” describes the model’s predicted coverage. The selector had no reviewed sack locations. Each start point could move at most once. The 126 moves ranged from 19.0 to 387.1 pixels, with median 124.0 pixels.

The protocol and executable were frozen before generating relocation outputs. All predictions were saved before reading labels for scoring. The run reused the 65 saved pairs of FIRC/CounTR outputs and made zero new network forwards or optimizer updates. The primary was chosen in advance. The FIRC-only control stopped under its own strictly beneficial exchange rule and consequently made fewer moves. The three matched controls used the same number of exchanges as the primary on each frame. Frozen protocol

## Why coverage failed to predict success

At each exchange we rescored one-to-one matching after removing the point and again after inserting its replacement. Removals destroyed **44** matches and replacements recovered **30**, a net loss of **14**. March 12 lost 39 and recovered 24. February 27 lost five and recovered six. Just 12 of the 126 removed points were within eight pixels of another current prediction. Fifty were within eight pixels of some reviewed sack before removal. Low unique model support was an unreliable indicator that a prediction could be discarded.

With the same removed points, selecting distant destinations by FIRC confidence produced **76** matches, 15 more than the primary coverage rule. Coverage using only FIRC produced 78 overall, with 73 on March 12 and five on February 27. The apparent pooled improvement for that secondary control came with a loss on the second recording group. None of the tested variants beat both saved baselines in both groups.

For the primary, 4-, 8- and 16-pixel matches were 24, 61 and 113, versus 33, 75 and 118 at the starting positions. Development ended with 152 false positives and 294 missed sacks. Four of the 23 frames gained matches relative to the 75-match control, ten tied and nine lost matches. On dev_02, for example, 42 predictions yielded nine matches after reallocation, down from 16. On dev_05, the same rule improved five matches to nine at 17 predictions. Thirteen of 18 positive frames contained a match, down from 14.

## Protections and paper claim

- MAE remained 7.478 sacks per frame. Every individual count and absolute count error agreed with both saved controls.
- On the three low-count frames all arms emitted zero points and missed all 11 sacks. Low-count MAE and GAME1/2 remained 3.667 and F1 remained zero. All five empty frames retained zero false positives. These protections follow from the inherited zero counts.
- Overall GAME1 worsened by 0.174, exceeding the frozen 0.1 allowance. GAME2 remained within the protection margin. On training frames the primary produced 286 matches versus 312 for the starting center refinement.
- The primary missed the pooled practical-effect margin, the requirement to improve within both recording groups and the margin over all matched-swap controls.

**Paper-ready finding:** On two reused development recording groups, increasing the joint spatial coverage of two frozen counters at fixed predicted counts exchanged 44 correct matches for 30 new ones. Confidence-ranked destinations recovered 15 more matches with exactly the same removals. These results establish a failure mode of this coverage objective on this cohort. They do not support a claim that relocation improved localization on new recordings. The five reserved recording groups remained unscored.

Independent reconstruction verified the coverage objective using direct density-window sums, all candidate exchanges, unchanged per-frame counts, one-to-one matching at 4/8/16 pixels and spatial grid errors for all 455 method-frame records. Results, per-frame metrics and move-by-move audit · Verification receipt

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

