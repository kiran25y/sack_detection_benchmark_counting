# Counting and localizing visible sacks in truck-loading recordings

[Download the complete benchmark with the 116 original frames](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/visible-sack-counting-v1.zip). The owner confirmed that the original frames were cleared for public release on November 17, 2026. Downstream reuse terms for this component are unspecified. Extract the ZIP to use the original image paths. The annotations, scoring script, exemplars and saved baseline predictions are available directly in this folder.


This benchmark pairs native-resolution images of loading activity with reviewed centers for distinguishable filled sacks. Count the sacks on the selected foreground truck and those being carried by its associated workers. Folded bags, background trucks, unattended conveyor material and fully hidden stock fall outside the target definition. Each center represents one visible instance in one frame. An instance can appear again in another frame.

| Split | Recording groups | Frames | Annotated sack centers |
|---|---:|---:|---:|
| Training | 3 | 42 | 338 |
| Development | 2 | 23 | 355 |
| Test | 5 | 51 | 644 |
| **Total** | **10** | **116** | **1,337** |

## Images and annotations

`frames.json` lists each image ID, file, SHA-256 hash, native dimensions, split, recording group and coarse foreground region. Related recordings and frames were grouped before assigning splits. Recording groups describe dependencies within the archive. Multiple frames in a group can show the same load. The benchmark measures transfer across these groups, with site-level generalization still unmeasured.

`annotations/train.json`, `annotations/development.json` and `annotations/test.json` map each image ID to its reviewed count and its list of native pixel centers in `[x, y]` order. The top-left pixel center is `[0, 0]`. Each count equals the length of its point list. The test set was annotated blind to the three frozen systems' predictions. An empty test frame has an explicitly confirmed count of zero. The fixed inference region is `[left, top, right, bottom]` in the native pixel-edge coordinate system. All reviewed centers count towards evaluation, including two test centers outside that region. The two optional exemplar crops in `references/` come from training images. Their source image IDs and boxes are in `references/manifest.json`.

## Scoring

Supply a JSON object mapping every frame ID in the requested split to a list of predicted centers in the original image coordinates. Use an empty list for a predicted count of zero. With Python 3 and NumPy installed, run this command from the extracted directory.

```text
python score.py --split test --predictions baselines/verifier88.json
```

MAE is the mean absolute count error across frames. Localization uses maximum-cardinality one-to-one matching within eight pixels after scaling the image long side to 1,024 pixels. F1 pools true positives, false positives and false negatives across frames. GAME1 and GAME2 sum the absolute count errors in 2×2 and 4×4 bins of the fixed foreground region, respectively, and average over frames. Points outside the region are assigned to its nearest edge bin for GAME. The scorer reports overall, per-group, low-count (1–11 sacks) and empty-frame results.

## Reference results

| Frozen method | Test MAE ↓ | Matches out of 644 | F1@8 ↑ |
|---|---:|---:|---:|
| Voting baseline | 6.12 | 150 | .3055 |
| Voting plus supplemental verifier | 5.06 | 167 | .3175 |
| Raw ranking at the verifier's per-frame count | 5.06 | 165 | .3137 |

The verifier lowered MAE in all five test groups. Raw ranking and the verifier use identical per-frame counts by design, so their difference measures point placement. The test set contains 29 low-count frames with 152 sacks and one empty frame. The verifier localized 31 of those 152 sacks and returned zero on the empty frame. The complete reference outputs and metrics are in `baselines/`. Use the development split to select model settings and thresholds.

## Availability and reuse

The project owner confirmed permission to release this benchmark on November 10, 2026. Downstream reuse terms are currently unspecified. The images show loading-site workers and vehicles. The annotations describe sacks and do not provide worker identities. The package preserves the image bytes used for evaluation. `SHA256.json` in the ZIP lists the checksums of its data, documentation and scorer.
