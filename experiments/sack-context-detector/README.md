# A standalone point detector using complementary image context

The new detector achieved mean development MAE **14.19** and point F1@8 **0.061** across three seeds. The ordinary point head on the same frozen DINOv2 encoder achieved **11.06** and **0.161**. The proposed architecture lost on both measures at each seed. All five empty frames had zero detections. All eleven sacks on the three low-count frames were missed at the frozen operating points.

## Architecture

The detector takes a single RGB frame and the existing foreground truck ROI. On the 896-pixel input, it groups ViT patches into 8×8-patch blocks in a checkerboard. DINOv2 processes the intact input and two complementary views made with its native mask token. A fourth, fully masked feature grid depends only on valid and padded geometry and can be cached. The head proposes point locations directly over the full grid. It takes no proposals from the earlier sack detectors.

At each cell the same convolutional head computes scores for the full image, the view with the local block visible, the complementary view with it hidden, and the fully masked grid. These four scores are denoted by F, K, H and N. The detector measures the change when it reveals the local block under two surrounding contexts. The two contrasts are F−H and K−N. The output logit is a learned positive weighted sum of those contrasts, plus a bias, minus learned penalties for disagreement between the contrasts and between the two visible-view center offsets. A softmax over the two contrasts weights the offsets. Masking uncovers the contribution of groups of patches, which can contain sacks, workers and truck structures together.

The comparison head receives all four views. It learns four unrestricted score-fusion weights and a bias. Both heads have **51,032 trainable parameters**, including the shared point head. The ordinary DINOv2 control uses one view and **51,027** trainable parameters. The four-view variants require three image-dependent encoder passes and a cacheable fully masked pass. Across 65 images and two valid geometries, the shared feature extraction took 197 passes. Architecture source · Frozen protocol

## Matched test

Each seed used the same initial point-head weights, frame order, point and offset supervision, optimizer, 420 updates, 64-candidate decoder, eight-pixel suppression radius and training-only cutoff rule across arms. We reused the completed ordinary seed-1729 checkpoint and reproduced its development scores and coordinates exactly. The eight other heads completed 3,360 updates. All thresholds were frozen before any new development predictions. The fitting split contains 42 reviewed frames across three groups. The development split contains 23 previously studied frames across two groups. Five reserved groups were excluded.

| Seed | Ordinary MAE | Fusion MAE | Proposed MAE | Ordinary F1@8 | Fusion F1@8 | Proposed F1@8 | Proposed matches / 355 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 11.78 | 14.61 | 14.00 | .139 | .053 | .088 | 17 |
| 1730 | 11.74 | 14.43 | 14.70 | .175 | .032 | .043 | 8 |
| 1731 | 9.65 | 14.43 | 13.87 | .169 | .032 | .051 | 10 |
| Mean | **11.06** | 14.49 | 14.19 | **.161** | .039 | .061 | 11.7 |

The proposed head beat ordinary four-view fusion on localization F1 at all three seeds and on MAE at two seeds. Its mean GAME1/GAME2 were 14.30/14.62, compared with 11.93/12.88 for the single-view control. On March 12 it matched 1, 0 and 1 sacks across the three seeds. On February 27 it matched 16, 8 and 9. The frozen backbone, matched-architecture and application gates all failed. A second implementation of point matching and spatial binning checked the saved outputs. Complete results and per-frame predictions · Run verification · Trained heads and fitting-only thresholds

## Contribution and paper decision

The specific design tested here is a dense point detector that uses two complementary patch-reveal contrasts for both scoring and subcell localization. DINOv2 supplies the frozen representation. Earlier work on density counting used point-derived masks and foreground distillation, and earlier counterfactual-attention work compared evidence under attention interventions. Generic masking or contrastive evidence alone would carry little novelty. This study does not establish priority for the precise combination or a performance advantage over its backbone. [DINOv2](https://arxiv.org/abs/2304.07193) · [Focus for Free in Density-Based Counting](https://ivi.fnwi.uva.nl/vislab/publication/zenglin-ijcv-2024/) · [Counterfactual Attention Learning](https://openaccess.thecvf.com/content/ICCV2021/html/Rao_Counterfactual_Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Re-Identification_ICCV_2021_paper.html)

The previous recording-included supplemental verifier achieved **88 matches, MAE 6.35, F1 .293 and zero points on all five empty frames**. Its 75-point-match voting control scored MAE 7.48 and F1 .264. Both systems missed the low-count sacks. The verifier adds an extra proposal source and was a comparison arm in an earlier experiment. It remains a concrete engineering candidate for independent confirmation. The current mask-based architecture does not supersede it. Earlier verifier and comparison

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 13 files under the folder sack-context-detector/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
