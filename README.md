# Counting and localizing visible sacks

This repository contains the experiments, annotations and numerical outputs for a recording-grouped study of visible sacks in truck-loading footage. A missed sack and a point on background can cancel in a frame's total. The point labels and matched-count comparisons let readers measure those errors separately.

## Datasets

| Dataset | Contents | Download |
|---|---|---|
| [Visible-sack point benchmark](datasets/visible-sacks/) | 116 native images, 1,337 reviewed centers, 10 recording groups. Train / development / test use 42 / 23 / 51 frames. | [Original images, labels, scorer and baselines](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/visible-sack-counting-v1.zip) |
| [Roboflow carried-sack boxes](datasets/roboflow-carried-sacks/) | 24,007 images and 16,400 boxes. Original export declares CC BY 4.0. | [Original YOLO dataset](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/Sackdetection.v1i.yolo26.zip) |
| [Reviewed whole-image totals](datasets/reviewed-totals/) | 300 counts joined to Roboflow images by archive path and SHA-256. | [JSON Lines labels](datasets/reviewed-totals/approved_whole_image_counts_300.jsonl) |

A zero-box label concerns the carried-sack task. It does not establish an empty truck. Each component has its own annotation scope and reuse information.

## Experiments

- [Browse the 31 sack experiments, controls and diagnostics](experiments/README.md). Each has a separate folder. The sequence covers IPIC, CounTR, FIRC, supplementary proposals, the blind verifier, mixed supervision, CountGD++ and P2PNet.
- [Browse 14 earlier public-data counting studies](public-counting-studies/README.md). Their scores use separate data and splits.
- [Download saved experiment outputs](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip). Inside the ZIP, the top-level folders correspond to the directories under experiments. The local scientific outputs include prediction arrays, scores, training records and collected research heads where available.
- [Download the four P2PNet score and coordinate banks](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/banks.tar). The tar contains 296 saved model-frame arrays across development and test.
- [Check file sizes and SHA-256 digests](release-assets.json).

## Results and replay

The originally blinded verifier reduced test MAE from 6.12 to 5.06 sacks per frame. At identical counts the verifier matched 167 centers and raw ranking matched 165. In later comparisons on the same 51 frames, the verifier matched 213 of 644 reviewed centers at a 625-point budget. At exactly those per-frame counts, the three adapted P2PNet seeds matched 90, 65 and 59. One-to-one matches use an eight-pixel tolerance at a full-image long side of 1,024 pixels.

For the [portable CPU replay](experiments/cross-seed-matched-budgets/replay/), download banks.tar into its full_factorial/ subdirectory. From the replay directory, run the following with Python 3.10+, NumPy 1.22+ and SciPy.

    python -B verify_review.py

To reproduce the original visible-sack baseline from the dataset folder:

    python score.py --split test --predictions baselines/verifier88.json

Third-party model weights and upstream source datasets used in the earlier public-data studies retain their original provider terms. Experiment folders identify their protocols, models and supervision. Benchmark imagery and auxiliary data use the terms in their respective dataset cards.
