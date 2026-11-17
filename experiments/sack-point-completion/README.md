# Can a detector recover sacks missed by the first pass?

**Decision, November 9. The experimental detector failed the bounded development test.** It matched four of 355 reviewed sacks, with all four matches on February 27. The strongest previous point control matched 75. Full training is unsupported.

## Detection method and fair control

We trained a 74,068-parameter detection head on a frozen pretrained DINOv2 ViT-B/14 with registers. It reads the foreground truck ROI and a spatial map of supplied point proposals. A verification branch scores existing points. A dense branch scores every valid feature cell and regresses a within-cell sack center. This permits a first detection on frames that receive zero initial proposals. Every fitting sack had its own feature cell. At deployment the head uses the image, the ROI and the first-pass proposals.

Both arms saw the same three views per update. Two contained the same number of marks. One marked a reviewed sack and the other moved that mark onto nearby background. The third alternated between an empty proposal set and the real first-pass proposals. Both arms received supervision for retained proposals and missing centers. The experimental arm also received a loss on the change in missing-center scores between the two equally sized sets. The underlying image pixels stayed identical across the views.

Each arm used the same initial weights, optimizer, 420 updates and frame order. The 42 fitting frames contained 338 reviewed sacks in three recording groups. Each head's additional-point threshold was selected on fitting data by minimizing empty false positives, maximizing positive-frame macro F1@8, minimizing MAE, then preferring the higher threshold on ties. Both thresholds were frozen before development decoding. They were 0.60 for ordinary supervision and 0.70 for paired supervision. The threshold for retaining a supplied point was 0.50 in both arms.

## Results on the same 23 development frames

A match is one-to-one within eight pixels after scaling the image long side to 1,024 pixels. GAME1 and GAME2 measure absolute count errors across 2×2 and 4×4 ROI subdivisions. Lower is better for both.

| Point method | Predicted | Matched | False positive | Missed | F1@8 | MAE | GAME1 | GAME2 | March 12 matches | February 27 matches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original point control | 213 | 71 | 142 | 284 | .250 | 7.48 | 8.70 | 10.17 | 62 | 9 |
| Voting control | 213 | 75 | 138 | 280 | .264 | 7.48 | 8.70 | 10.09 | 68 | 7 |
| New head, ordinary supervision | 30 | 9 | 21 | 346 | .047 | 14.13 | 14.13 | 14.39 | 3 | 6 |
| New head, paired supervision | 7 | 4 | 3 | 351 | .022 | 15.13 | 15.13 | 15.13 | 0 | 4 |

All four methods returned zero on every empty frame and missed every sack on the three low-count frames. Those frames contained 2, 2 and 7 sacks, all from March 12. Their MAE was 3.67. February 27 had four empty frames and four positive frames. On its positive frames the paired detector predicted 3, 1, 0 and 3 sacks against counts of 36, 18, 16 and 16, yielding 3, 0, 0 and 1 matches. March 12 had 269 sacks across 15 frames and February 27 had 86 across eight.

An adapted CounTR density baseline previously scored MAE 7.67, GAME1 10.17 and GAME2 13.71 on these same frames using a different training budget. It predicted a mean of 5.82 sacks per empty image. Its density maps have no native discrete point F1. Saved CounTR comparison

## What failed

The paired detector discarded 206 of 213 supplied proposals and produced zero additional points after decoding. Its seven retained proposals accounted for all four matches. Keeping every original voting proposal and adding the paired detector's accepted new points would reproduce the 75-match control exactly. The ordinary detector retained 27 supplied points and emitted three more. The three additions yielded zero additional one-to-one matches. The paired term therefore had no positive effect over matched ordinary supervision on the primary endpoint. Empty protection was the only frozen gate component that passed.

The first pass covered 312 of 338 fitting sacks (92.3%) and only 75 of 355 development sacks (21.1%). Consequently its real fitting proposals left only 26 genuine misses with which to learn recovery, versus 280 misses on development. Synthetic omissions increased the fitting signal, but the resulting head recovered none of the missing development sacks at its frozen operating point. On fitting frames the paired head produced 16 new points and matched 272 sacks in total. It already trailed the original voting proposals there. The proposal shift is measured. This experiment does not isolate its causal contribution from appearance shift or calibration.

## Paper decision

The candidate contribution was a paired loss for the spatial change in missing-center predictions when a correct point is replaced by a misplaced one. Prior detectors already condition on detection history and use contrastive denoising. These results provide no basis for claiming a superior detector or a novel general solution. [IterDet](https://arxiv.org/abs/2005.05708) · [DINO detection transformer](https://arxiv.org/abs/2203.03605)

This was one seed on two repeatedly studied development recordings. Five reserved recording groups were excluded. Before investing in another detector, obtain first-pass predictions on fitting recordings from a model that never trained on the corresponding recording. Measure whether the resulting real misses cover the sparse and occluded cases that a completion head is expected to recover. A future method claim would need a gain on untouched recordings against a strong pretrained detector with the same inference inputs.

The verified DGX results include both checkpoints, all 840 updates, both fitting-only calibration curves, all predictions and per-frame independent scoring. The container exited successfully without a guard stop. Per-frame results and gate checks · Run verification · Frozen protocol

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 8 files under the folder sack-point-completion/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
