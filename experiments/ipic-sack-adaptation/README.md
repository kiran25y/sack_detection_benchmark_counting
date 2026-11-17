# IPIC sack adaptation: verified A–F result, November 4, 2026

All six arms completed 1,000 saved updates each, with exit0, no OOM and no guard stop. Total 6,000 updates/48,000 pairs/96,000 query presentations. The final container finished at approximately 2026-11-04 07:12 KST after 17.99 hours. DGX timestamps were corrected using the observed host/client offset; this is shared-host elapsed time.

All output/release/input hashes, sequential update and augmentation coverage, saved checkpoint literal metadata, and the unchanged foundation/draw identities passed. Model/Adam tensor finiteness was checked by the frozen worker. This verification reads hashes, literal metadata and saved predictions without deserializing tensors or executing a model. All 30 saved evaluations were independently recomputed, including exact 11-query truth, development threshold selection, count/GAME1/2 and maximum-matching F1 at 4/8/16. A/F final count metrics received a separate stdlib spot check.

## Final update 1000 results

These are 11 natural sack development frames from **one recording group**, not the 1,286-query FSC cohort or an independent test. All frames are below 150 sacks. Each threshold selected below is selected on this same development cohort; its accuracy is optimistic for future data. No checkpoint or threshold outside the frozen protocol was chosen.

| Arm | Configuration | MAE at 0.50 | Selected threshold | Selected MAE | RMSE | Bias | GAME1 | GAME2 | F1@8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | B0 independent | 22.1818 | 0.10 | 10.8182 | 13.8269 | -0.8182 | 16.0909 | 23.7273 | 0.0859 |
| B | B0 global + local | 22.0000 | 0.10 | 11.3636 | 13.0210 | +2.8182 | 15.1818 | 22.8182 | 0.1033 |
| C | B1 independent | 22.1818 | 0.15 | 13.9091 | 15.6176 | -13.5455 | 14.8182 | 18.8182 | 0.0756 |
| D | B1 global + local | 22.0909 | 0.10 | 11.1818 | 12.5082 | +1.0000 | 13.5455 | 21.1818 | 0.1012 |
| E | B0 global only | 22.0000 | 0.10 | 10.5455 | 12.7778 | +0.1818 | 14.1818 | 21.0909 | 0.1046 |
| F | B1 global only | 22.0000 | 0.10 | 10.3636 | 13.0105 | +2.3636 | 13.8182 | 22.0000 | 0.1138 |

At 0.50, A/C predict zero for every image; B/E/F predict two sacks in total and D predicts one, against 244 annotated sacks. No predicted point in any positive image matches truth within 8 pixels at the normalized 1024 scale. The macro F1@8 of 0.0909 comes entirely from the one correctly empty image. The selected thresholds increase count recall but localization remains weak; positive-image F1@8 ranges 0.0832–0.1252. Five arms select the lower grid endpoint 0.10; no extra threshold search is authorized.

## Prespecified contrasts

| Candidate−control | Threshold policy | ΔMAE | ΔRMSE | ΔGAME1 | ΔGAME2 | ΔF1@8 | Relative MAE reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| B−A | common_0.50 | -0.1818 | -0.1640 | -0.1818 | -0.1818 | +0.0000 | +0.82% |
| B−A | development_selected | +0.5455 | -0.8059 | -0.9091 | -0.9091 | +0.0175 | -5.04% |
| D−C | common_0.50 | -0.0909 | -0.0857 | -0.0909 | -0.0909 | +0.0000 | +0.41% |
| D−C | development_selected | -2.7273 | -3.1094 | -1.2727 | +2.3636 | +0.0256 | +19.61% |
| B−E | common_0.50 | +0.0000 | +0.0460 | -0.1818 | -0.1818 | +0.0000 | +0.00% |
| B−E | development_selected | +0.8182 | +0.2431 | +1.0000 | +1.7273 | -0.0013 | -7.76% |
| D−F | common_0.50 | +0.0909 | +0.0821 | +0.0909 | +0.0909 | +0.0000 | -0.41% |
| D−F | development_selected | +0.8182 | -0.5023 | -0.2727 | -0.8182 | -0.0127 | -7.89% |

Lower signed error differences are better; higher F1 differences are better. B−A improves common-threshold MAE only 0.82%, below 5%, and worsens selected-threshold MAE 5.04%. D−C improves common-threshold MAE only 0.41%. Its selected-threshold MAE improves 19.61%, but GAME2 worsens 2.3636 (12.56%); the empty frame rises from 2 to 12 false detections. The thresholds differ for C and D, so this contrast combines learned outputs and the prescribed development calibration.

**The required local-versus-global MAE benefit is absent:** B−E and D−F each worsen selected-threshold MAE by 0.8182. B also worsens RMSE, GAME1/2 and F1 relative to E. D improves RMSE/GAME1/2 relative to F but worsens MAE and F1; its empty-image false detections rise from 7 to 12. Do not promote F merely because its selected MAE 10.3636 is smallest; it is a control, and its selected threshold is evaluated on the same development group.

## Decision and limits

The engineering run completed successfully. The available natural-development evidence does not support the proposed local-loss contribution. The complete scientific gate remains **unevaluable and not passed**, because independently verified physical intervention pairs and an independent final test are absent. Dense/extreme sack gains cannot be tested with this 0–34-count cohort. Do not start confirmatory training or claim novelty/independent accuracy from these results.

The saved outputs establish low confidence at 0.50 and poor localization at the allowed lower thresholds. They do not yet identify whether limited recording diversity, object scale/domain shift, inadequate optimization or assignment/localization learning is the dominant cause. Preserve the six checkpoints and the public foundation; do not change losses or thresholds based on a favorable subset.

Natural sacks remained point-only; no uncertain sack masks were cutouts. All arms received the same full image, fixed operator camera scope, training-only references and frozen draws. D2S composites are synthetic products on a training background, not observed physical sack interventions. Public D2S/FSC category overlap prevents strict unseen-category claims.

Detailed receipt: `study_protocols/ipic_sack_adaptation_completion_20261104.json`. Full per-image evidence: `artifacts/ipic_sack_screen_v1_20261103/completion_analysis.json`. Original launch and failed preparation evidence remain preserved.

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 102 files under the folder ipic-sack-adaptation/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
