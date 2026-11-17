# Where sack localization failed and what CounTR recovered

**November 8, 2026.** Adding weak FIRC candidates supported by the CounTR-P density map reduced development MAE from 11.826 to 7.478. One-to-one matches increased from 25 to 72 of 355. At identical predicted counts per image, FIRC confidence alone located 71 sacks. The registered density map failed the prespecified test for a distinct localization benefit. Practical readiness also failed.

## Why training MAE understated the error

The 0.929 training MAE was measured on the 42 reviewed frames used for 12,000 sack presentations during adaptation. Those frames came from three recording groups. The 23 development frames came from two other groups. Positive-image macro F1 fell from 0.953 on training to 0.089 on development.

For each reviewed center, we inspected all saved candidate points within eight pixels in the long-side-1024 chart. The median highest score there fell from **0.577 on training to 0.178 on development**. All 355 development centers had a candidate inside the matching radius. At 330 centers every such candidate scored below the frozen 0.35 threshold. A dense candidate grid makes geometric coverage easy. The scores determine whether an instance is detected. The median highest nearby candidate still ranked at the 99.78th percentile among its own frame’s candidates, a descriptive measure that uses the known center to choose the neighborhood.

FIRC emitted 83 points for 355 sacks. Of those, 25 matched, 58 were false positives and 330 sacks were missed. Of the 83 accepted points, 32 lay within eight pixels of some truth point. One-to-one matching could credit 25 of them. The other 51 lay outside every annotated eight-pixel neighborhood. Widening the audit radius to 16 and 32 pixels yielded 39 and 64 matches. The shortage of accepted points and their placement both matter. Lighting, viewpoint, packaging and recording group vary together here, so the audit cannot identify their individual causal effects.

## Failed frames

| Frame | Reviewed sacks | Original FIRC count / matches | CounTR-P count | Tested count / matches |
|---|---:|---:|---:|---:|
| dev_04, red stacked load with glare and workers | 30 | 2 / 0 | 29.5 | 13 / 5 |
| D27_04, pale sacks under a bright vertical glare band | 36 | 11 / 5 | 12.9 | 16 / 6 |
| D12_00, two sacks at the back of the bed | 2 | 0 / 0 | 15.9 | 0 / 0 |
| dev_00, empty bed with workers | 0 | 0 / 0 | 16.2 | 0 / 0 |

On dev_04 the CounTR map concentrates on much of the loaded area. On D12_00 and dev_00 it also assigns density to the blue truck panel and other non-target image regions. A predicted total close to the reviewed count can coexist with poor instance localization.

Side-by-side frame evidence · Paired results for all 23 frames

## One controlled test

For the 330 misses, the median local CounTR density around the best weak candidate near a known sack was 3.87 times the image ROI mean. Rotating the map 180 degrees about the ROI center gave a median of 0.94. Correct registration gave higher density in 281 of 330 such comparisons. These label-assisted observations motivated the experiment. Inference uses only the two model outputs and the original coarse camera ROI.

The test preserved the original points accepted at 0.35. For the weaker candidates it used a common eight-pixel duplicate-suppression rule and a fixed eight-pixel local-density window. It multiplied candidate odds by the local support factor clipped to [0.25, 4]. Each readout used the same cutoff grid and tie rule on 42 training frames. Cutoffs were recorded before computing the new development predictions. Each control used the same candidate bank. The point-only control removed density weighting, the total-count control spread the same predicted density uniformly, and the rotated control misplaced the same density field. A second comparison fixed the number of proposals added on every image and varied their ranking. The primary method supplied that number using its model scores.

The window, multiplier bounds and decision margins were experimental choices frozen for this check. Existing endpoint predictions cover all frames. The run used zero new network forwards, optimizer calls or reserved images. This tests the decoder operating on those saved predictions. It does not establish how the result would change with joint training. Frozen protocol

| Development method | Weak-candidate cutoff | MAE | Pooled F1@8 | ROI GAME1 / GAME2 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Original FIRC | — | 11.826 | 0.1142 | 11.826 / 12.348 | 25 / 58 / 330 |
| CounTR-P density | — | 7.670 | — | 11.284 / 15.047 | — |
| Point-only recovery | 0.30 | 8.478 | 0.2103 | 9.087 / 10.391 | 55 / 113 / 300 |
| CounTR total-count weighting | 0.55 | 9.739 | 0.2069 | 10.000 / 11.217 | 54 / 113 / 301 |
| Rotated density weighting | 0.60 | 10.739 | 0.1296 | 11.087 / 11.696 | 30 / 78 / 325 |
| **Aligned density weighting** | **0.60** | **7.478** | **0.2535** | **8.609 / 10.000** | **72 / 141 / 283** |
| Same number, rank by point confidence | Per-image number fixed | 7.478 | 0.2500 | 8.696 / 10.174 | 71 / 142 / 284 |
| Same number, rank by total-count weighting | Per-image number fixed | 7.478 | 0.2500 | 8.696 / 10.174 | 71 / 142 / 284 |
| Same number, rank by rotated support | Per-image number fixed | 7.478 | 0.2183 | 9.043 / 10.609 | 62 / 151 / 293 |

CounTR-P retains its native fractional density integral. Its original spatial mass precision and recall were 6.81% and 7.48%. Discrete point F1 is undefined for that output without an additional point extractor. We recomputed its GAME1/2 on the same ROI bin boundaries used for FIRC. The historical CounTR report used whole-image bins. FIRC and CounTR had different original training budgets, and the combined method consumes both models at deployment.

## Where gains and costs occurred

The aligned readout added 130 predictions across development. Matches increased by 47 and false positives increased by 83. Its MAE fell by 4.348 sacks per frame (36.8%). RMSE fell from 15.240 to 10.013 and signed bias changed from −11.826 to −6.174. Positive-image macro F1 rose from 0.089 to 0.202. Fourteen of 18 positive frames had at least one match.

| Development subset | Original FIRC | CounTR-P | Aligned readout |
|---|---:|---:|---:|
| Low-count MAE, three frames | 3.667 | 12.804 | 3.667 |
| Mean predicted count on five empty frames | 0 | 5.824 | 0 |
| MAE on 15 frames with 12–36 sacks | 17.400 | 7.259 | 10.733 |
| March 12 group MAE, 15 frames | 13.867 | 8.007 | 7.600 |
| February 27 group MAE, eight frames | 8.000 | 7.039 | 7.250 |

Both point readouts missed all 11 sacks on the three low-count frames. The aligned readout’s small overall MAE advantage over CounTR-P was 0.192. Its absolute error on higher-count frames was about 52 sacks worse in total than CounTR-P. Lower error on low-count and empty frames offset that difference. The two recording groups also give different orderings of the two methods.

## Decision for the paper

The aligned readout passed the planned MAE and F1 gains over original FIRC and protected both recording-group MAEs, overall spatial grid errors, low-count metrics and empty frames. Its F1 gain over the count-matched rotated control was 0.0352. Against each of the two simpler count-matched rankings the gain was only **0.0035**, below the prespecified 0.02. Of 130 added points, 118 had the identical proposal ID under the aligned and point-confidence rankings. Only six frames changed their added proposal set.

The absolute MAE criterion passed. Pooled F1 (0.254), positive-image macro F1 (0.202), low-count MAE (3.667) and low-count F1 (0) failed the practical criteria. All five empty frames remained correct. Training positive-image macro F1 was 0.965.

**Defensible finding:** Admission of more weak point candidates reduced count error on this development cohort. At matched counts per frame, registered density guidance produced one additional net eight-pixel match over confidence ranking. The data support reporting a count-versus-localization ablation. They do not demonstrate a distinct localization mechanism or establish deployment accuracy. The reused development labels informed the hypothesis, and the five reserved recording groups remained unscored. The two-model compute cost is also part of this comparison.

Density-guided detection and point extraction from predicted density have published precedents. Generic density guidance is insufficient for a priority claim. [Lian et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Lian_Density_Map_Regression_Guided_Detection_Network_for_RGB-D_Crowd_Counting_CVPR_2019_paper.html) · [Lin and Chan, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Optimal_Transport_Minimization_Crowd_Localization_on_Density_Maps_for_Semi-Supervised_CVPR_2023_paper.html)

Independent matching and spatial-grid arithmetic reproduced all 520 method-frame records. The baseline reproduced all original per-frame outputs, and every equal-number control preserved the primary count on every frame. Full experiment result · Verification · Candidate-level failure audit

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

