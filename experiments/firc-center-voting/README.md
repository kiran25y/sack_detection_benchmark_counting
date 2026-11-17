# Sack center refinement at fixed counts

**On 23 development frames, confidence-weighted local averaging produced 75 correct locations from the same 213 predictions that gave the confidence-ranking control 71.** The previous CounTR-aligned ranking had 72. The four-match improvement fell short of the prespecified six-match minimum. March 12 gained six matches and February 27 lost two. The experiment therefore failed its criterion for improvement across recordings.

## Comparison

All methods started with the exact per-frame counts and seed identities of the November 8 confidence-ranking control. That control inherited each frame’s count from the previously frozen CounTR-assisted readout. The present experiment changed only point coordinates. The evaluation used maximum one-to-one matching in the same native-image chart, scaled to a long side of 1,024 pixels.

| Development readout | Matches at 4 px | Matches at 8 px | Matches at 16 px | Pooled F1 at 8 px | March 12 matches at 8 px | February 27 matches at 8 px |
|---|---:|---:|---:|---:|---:|---:|
| Saved confidence ranking | 31 | 71 | 113 | 0.2500 | 62 | 9 |
| Average nearby candidate centers equally | 24 | 66 | 116 | 0.2324 | 58 | 8 |
| **Weight nearby centers by FIRC confidence** | **33** | **75** | **118** | **0.2641** | **68** | **7** |
| Move equally far in the opposite direction | 23 | 70 | 111 | 0.2465 | 60 | 10 |

The original 71-match control had 79 predictions within eight pixels of any reviewed center. One-to-one matching credited 71. The remaining 134 predictions lay outside all eight-pixel neighborhoods. Expanding the diagnostic radius from eight to 16 pixels increased the baseline matches from 71 to 113, which motivated testing the coordinates of near misses.

For each training frame with two or more sacks, we measured the median distance from each reviewed center to its nearest neighbor. The median across those frames was 27.60 pixels. We used half that distance, 13.80 pixels, as the local voting radius. Each saved FIRC candidate inside a seed’s neighborhood voted for its nearest seed. The proposed update moved the seed toward the average candidate location weighted by the original FIRC confidence. Each seed received one update, capped at eight pixels. Median movement was 1.57 pixels and the largest movement was 5.80 pixels. The uniform control used the same voters. The opposite-direction control matched the primary movement length for every point, including at ROI boundaries.

We froze the rule before computing its development outputs and used zero additional model forwards or optimizer steps. The two development recording groups had been used in earlier studies and supplied the descriptive tolerance audit. The five reserved recording groups had no approved exhaustive labels for this comparison and received no scores. Frozen test protocol

## What improved and what stayed difficult

- Pooled F1 improved by 0.0141 over the 71-match control. The frozen practical margin was 0.02. With 213 predictions and 355 reviewed sacks, that required at least 77 matches.
- Weighting beat uniform averaging by nine matches and 0.0317 F1. It beat moving in the opposite direction by five matches and 0.0176 F1. Only the uniform comparison cleared the 0.02 attribution margin.
- Five development frames gained matches, 16 tied and two lost matches. The largest gain was on **dev_02**, from 14 to 16 matches with 42 predictions and 33 reviewed sacks. The two losses occurred on **D27_04** (6 to 5) and **D27_07** (2 to 1). Fourteen of 18 positive frames still had a match. The final totals were **75 true positives, 138 false positives and 280 misses**.
- Count MAE stayed at 7.478 sacks per frame for every readout by construction. GAME1 stayed at 8.696 and GAME2 changed from 10.174 to 10.087 versus the 71-match control. Positive-image macro F1 rose from 0.200 to 0.208. The training cohort improved from 309 to 312 matches out of 338.
- Every low-count and empty-image protection passed. Both methods emitted zero points on the three low-count frames, missing all 11 sacks and giving low-count MAE 3.667 and F1 zero. All five empty frames had zero false positives. Fixing each frame’s count prevents this test from recovering the low-count sacks.

## A bound on what a local correction can accomplish

If a prediction moves at most eight pixels and then matches a reviewed center within eight pixels, its original location must have been within 16 pixels of that same center. The final matches still have distinct predictions and distinct centers. They therefore constitute a feasible matching for the original points at 16 pixels. That maximum matching had cardinality **113**.

Consequently **any** such local correction applied to these seeds has at most 113 eight-pixel matches and pooled F1 at most **226/568 = 0.398**. This is a cohort-specific geometric bound. The previous 0.50 pooled F1 readiness target would require 142 matches at the same counts. At least 29 of those matches would need predictions to reach beyond the local movement constraint. For the low-count frames a model must also emit candidates where the inherited count is zero.

These observations prioritize the choice of *which sack regions receive points*. A concrete next mechanism is to let a low-quality seed be reassigned to a distant, exemplar-supported candidate in an uncovered region. A future test would need to beat confidence ranking at identical per-image counts and demonstrate transfer on separately reviewed recordings. This experiment supplies a necessary geometric condition for that work. It gives no measured accuracy for the proposed reassignment.

## Wording supported for the paper

> On 23 previously used sack development frames from two recording groups, we held all 213 predicted instances fixed in number and refined their coordinates. Confidence-weighted local averaging increased eight-pixel one-to-one matches from 71 to 75 (pooled F1 from 0.250 to 0.264). Group changes were +6 and −2. An oracle bound derived from the unmodified coordinates limited any eight-pixel relocation of these seeds to 113 matches, below the 142 required for 0.50 pooled F1. The experiment identifies proposal coverage as a constraint on localization in this cohort.

The evidence supports a controlled error analysis. It falls short of a broadly generalizing localization improvement, a deployment result or a new voting algorithm. Confidence-weighted accumulation of local offset votes has precedent in keypoint localization. [Papandreou et al., *PersonLab*, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/George_Papandreou_PersonLab_Person_Pose_ECCV_2018_paper.pdf)

The independent verifier reconstructed all coordinates to within 1.4 × 10⁻¹² native pixels and reproduced the 4-, 8- and 16-pixel matches and GAME1/2 on all 260 frame-method records. Complete result and per-frame records · Independent verification

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

