# Can a local image patch distinguish a missed sack from a false detection?

**Verified November 10, 2026 at 13:19 KST.** The trained refinement head reduced MAE and increased pooled localization F1 over the frozen verifier on both recording splits. Its prespecified joint counting, localization and empty-truck criteria failed. At the same selected points, moving their coordinates gained two matches on development and zero net matches on the previously scored test.

## Training and comparison

The starting model had already trained on 21,006 Roboflow carried-sack images, 300 reviewed whole-image totals and 42 benchmark fitting frames. Its lower test count error came with 73 additional false detections and eight additional one-to-one matches. On the fitting recordings, 205 of 338 labeled centers lay within 8 pixels of an independently generated recording-excluded proposal and 298 lay within 16. These nearest-proposal counts can share proposals. They motivated a bounded relocation test. Previous study

Both new heads used the same frozen DINOv2 encoder, 42 sack fitting frames, reviewed foreground regions, initialization, local patches, AdamW schedule and sampling. Each head classified whether a sack center lay within 16 pixels. The primary head also learned a bounded 2D correction. All distances use a 1,024-pixel full-image long side. Three fits each excluded one entire training recording group. Their candidate scores set one threshold per head. A fourth fit used all 42 fitting frames. Each fit ran 250 updates per head for 2,000 verified updates in total. The original verifier points were preserved. Only additional joint-model candidates could move or be removed. Zero-box Roboflow frames supplied no empty-truck labels.

The proposal generator used to calibrate the thresholds differed from the joint-model generator used at evaluation. Thresholds were 0.8990 for confidence only and 0.8651 for refinement. The GPU worker received development and test images and frozen proposals. Their labels entered the separate local scoring step. Both splits had informed prior research decisions. Frozen protocol · Training verification

## Counting and localization

F1 uses pooled, one-to-one center matches within 8 pixels. GAME1 and GAME2 sum absolute count errors over 2×2 and 4×4 spatial grids, respectively. Lower values are better. The low-count subset contains 1–11 reviewed sacks per frame. There are 11 low-count sacks and five reviewed empty frames on development, and 152 low-count sacks and one reviewed empty frame on test.

| Split and method | MAE ↓ | F1@8 ↑ | Matches | False points | GAME1 / GAME2 ↓ | Low-count matches | False points on empty trucks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development · verifier | 6.35 | .2933 | 88 | 157 | 7.74 / 9.57 | 0 | 0 |
| Development · joint proposals | 6.57 | .2719 | 104 | 306 | 8.91 / 11.26 | 2 | 6 |
| Development · confidence only | 5.43 | .2929 | 93 | 187 | 7.35 / 9.17 | 0 | 0 |
| Development · **refined** | **5.61** | **.3126** | **98** | **174** | **7.52 / 9.35** | **0** | **0** |
| Development · same selected points, original coordinates | 5.61 | .3062 | 96 | 176 | 7.52 / 9.35 | 0 | 0 |
| Development · first joint proposals at equal count | 5.61 | .2935 | 92 | 180 | 7.61 / 9.43 | 0 | 0 |
| Test · verifier | 5.06 | .3175 | 167 | 241 | 5.80 / 6.67 | 31 | 0 |
| Test · joint proposals | 4.22 | .3089 | 175 | 314 | 5.55 / 6.73 | 38 | 2 |
| Test · confidence only | 4.55 | .3198 | 173 | 265 | 5.45 / 6.47 | 37 | 2 |
| Test · **refined** | **4.67** | **.3210** | **173** | **261** | **5.65 / 6.63** | **36** | **2** |
| Test · same selected points, original coordinates | 4.67 | .3210 | 173 | 261 | 5.65 / 6.63 | 36 | 2 |
| Test · first joint proposals at equal count | 4.67 | .3191 | 172 | 262 | 5.65 / 6.67 | 35 | 2 |

The refined head lowered MAE relative to the verifier in all seven development and test groups. Its two matches gained by moving the selected points on development came from the same recording. On test, moving those points gained one match on H071 and lost one on H044. Compared with raw joint proposals, refinement removed 53 false points on test and lost two matches. Both points on empty H025 exceeded the training-calibrated threshold, scoring .9390 and .9232.

## Benchmark interpretation

Filtering the additional proposals improved the tradeoff between count and localization relative to the original verifier on these recordings. Confidence-only filtering achieved lower MAE than refinement on both splits. The primary failed to retain the joint model's low-count matches and still predicted sacks on H025. The observations support a benchmark comparison of count error, localization and empty-truck behavior. An independent improvement attributable to moving point coordinates, or deployable empty-truck accuracy, remains unestablished. These data span one seed, three fitting groups, two development groups and five previously scored test groups. Per-frame results and decision checks · Trained refinement head

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 12 files under the folder sack-local-refinement/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
