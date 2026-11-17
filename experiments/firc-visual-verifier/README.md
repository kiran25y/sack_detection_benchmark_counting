# Why FIRC coverage differed by recording, and a controlled appearance test

**No tested fix beat both saved controls in both recordings.** The prespecified frozen-DINOv2 appearance verifier produced 48 eight-pixel matches on March 12 and eight on February 27. The confidence-ranking controls had 62 and nine, and the center-refined controls had 68 and seven. Holding every image’s count fixed, the visual verifier recovered three matches over FIRC-only coverage on February 27 and lost 25 on March 12.

## Why FIRC-only obtained 73 and five

| Recording | Reviewed sacks | Predicted points | Confidence ranking | After center refinement | Coverage swaps | Matches lost on removal | Matches gained on insertion | FIRC-only endpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| March 12, 15 frames | 269 | 185 | 62 | 68 | 61 | 16 | 21 | **73** |
| February 27, 8 frames | 86 | 28 | 9 | 7 | 12 | 3 | 1 | **5** |

Much of the gap predates the swaps. The February point budget allowed at most 28 detections for 86 sacks. Four of its eight frames were empty. March received 185 detections for 269 sacks. Placement also differed. March replacements acquired 21 matches in 61 attempts, versus one in 12 for February. Seven February destinations were within 16 pixels of a reviewed point, and all 12 were within 32 pixels. Only one was within eight. Bright vertical glare, close packing and people are visible in the February crops. These observations identify the error in the coordinates. They cannot isolate a causal effect of lighting or packing across two groups.

| Failed or informative frame | Sacks / predicted count | Matches at start | Swaps | Lost / gained | Final FIRC-only matches |
|---|---:|---:|---:|---:|---:|
| D27_04, loaded pale stack | 36 / 16 | 5 | 9 | 3 / 1 | 3 |
| D27_05, workers at stack | 18 / 3 | 0 | 2 | 0 / 0 | 0 |
| D27_06 | 16 / 5 | 1 | 1 | 0 / 0 | 1 |
| D27_07 | 16 / 4 | 1 | 0 | 0 / 0 | 1 |
| dev_04, red stacked load | 30 / 13 | 6 | 5 | 2 / 4 | 8 |
| dev_05, red stacked load | 21 / 17 | 5 | 3 | 0 / 2 | 7 |
| dev_08, glare over load | 18 / 24 | 5 | 7 | 3 / 1 | 3 |
| dev_09, glare over load | 19 / 13 | 5 | 8 | 3 / 1 | 3 |

The entire two-match February drop from centering to coverage occurred on D27_04. Three of the other February positive frames contributed a combined two endpoint matches. The median nearest-label distance of replacement points was 12.87 pixels on February and 10.70 on March. Reviewed exchanges overlaid on eight frames · All original moves and frame results

## One new test

The verifier asked whether image appearance could distinguish a sack center from another plausible but misplaced point. A single frozen DINOv2 encoder pass per frame provided a central patch descriptor and its difference from a ring of eight neighboring descriptors. The ring radius of 13.80 pixels came from the earlier training-only spacing estimate. A training gallery contained 338 reviewed centers and 679 hard negatives, including near misses and high-confidence locations away from the labels. Each score compared similarity to positive and negative examples from three distinct training frames. Training-frame diagnostics excluded the query’s entire recording group from the gallery. Development used only the training gallery.

The 75-match coordinates, eligible destination bank, maximum number of exchanges per frame, and point-count budget were identical to the earlier coverage experiment. The primary ranked removals and destinations by the appearance margin and could abstain from a swap. It accepted all 73 available development moves. The matched controls used those exact per-frame exchange counts. Cutoffs, frames and coefficients were fixed before scoring the new predictions. The 65 feature forwards completed with an unchanged encoder and clean Docker exit. There were zero gradient updates and zero reserved images scored. Frozen protocol

| Development readout | March 12 | February 27 | Matches / 355 | Pooled F1@8 | ROI GAME1 / GAME2 |
|---|---:|---:|---:|---:|---:|
| Saved confidence ranking | 62 | 9 | 71 | 0.250 | 8.696 / 10.174 |
| Saved center refinement | 68 | 7 | 75 | 0.264 | 8.696 / 10.087 |
| Prior FIRC-only coverage | 73 | 5 | 78 | 0.275 | 8.609 / 9.652 |
| **Primary visual center + ring** | **48** | **8** | **56** | **0.197** | **9.043 / 10.609** |
| Matched FIRC confidence ranking of exchanges | 71 | 7 | 78 | 0.275 | 8.435 / 9.826 |
| Matched FIRC-only coverage prefix | 73 | 5 | 78 | 0.275 | 8.609 / 9.652 |
| Matched center descriptor alone | 68 | 9 | 77 | 0.271 | 9.217 / 10.261 |
| Matched shuffled visual scores | 61 | 7 | 68 | 0.239 | 8.522 / 10.174 |

On March the primary removed 27 correct matches and gained seven. On February it removed three and gained four. On D27_04 it localized six sacks, versus three with FIRC-only coverage. On dev_02 it localized 12, down from 16 with center refinement. The center-descriptor control tied the stronger saved control on each recording. The ring comparison provided no evidence of additional localization value under this rule.

The primary ended with 56 true positives, 157 false positives and 299 missed sacks. There were 21 matches at four pixels and 111 at 16. Fourteen of 18 positive frames had a match. Training-frame positive macro F1 was 0.940 when the gallery excluded the entire query recording group, while development positive macro F1 was 0.155. The underlying FIRC model had optimized on those training frames, so this training statistic is descriptive.

## Decision

Strict improvement over both historical controls required at least 69 matches in March and ten in February. The primary failed both comparisons against those group-specific targets and missed the pooled six-match improvement over the 75-match control. Overall GAME1 and GAME2 also exceeded their nonregression allowances.

**Low-count and empty protections passed mechanically.** MAE was 7.478 across the same 213 predicted points for all methods. On the three low-count frames zero predictions missed all 11 sacks, giving MAE 3.667, F1 zero and GAME1/2 of 3.667. The five empty frames retained zero false positives. Fixing an inherited zero count prevents recovery on those sparse frames.

**Paper wording.** On two training-disjoint, previously reused development recording groups, the same FIRC-only coverage rule gained five matches in one group and lost two in the other relative to center refinement. The February loss was concentrated in one densely loaded frame. A training-only visual-neighbor verifier at the same per-frame counts and swap budget recovered three matches relative to FIRC-only in February and lost 25 in March. These results establish recording-dependent errors in the placement rule. They do not establish a localization improvement on new recordings or supply final deployment accuracy.

The independent verifier recomputed gallery similarities in FP64, all selected swaps and controls, one-to-one matching at 4/8/16 pixels and spatial-grid errors for 520 method-frame records. The largest similarity-score discrepancy was below 3.7 × 10⁻⁷. Result and per-frame metrics · Independent verification

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

