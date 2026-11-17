# Can real misses from other recordings teach a detector to recover sacks?

**Decision on November 9, 2026. The bounded test failed.** The primary verifier increased one-to-one development matches from 75 to 116 and lowered MAE from 7.48 to 5.65. It missed all 11 low-count sacks and added one false point to each of two empty trucks. At exactly the same per-frame count, the original proposal scores placed eight more points correctly.

## What the proposal models saw

The proposal head reads frozen DINOv2 features from the predeclared foreground ROI. It predicts a center score and an offset for each valid 14-pixel feature cell. Each of three freshly initialized heads trained for 420 updates on two fitting recording groups and predicted the third. Related clips stayed together. Each head chose its acceptance threshold using only its own fitting groups. The worker verified pretrained weights and image identities. No previously sack-adapted weights or previously fitted point predictions entered these folds.

A fourth head trained for 420 updates on all 42 fitting frames and supplied development candidates. These bounded heads are a new proposal source. The historical 71- and 75-match controls were reproduced from their saved predictions. Five reserved groups were excluded. Training proposals, native coordinates and fold identities · Frozen proposal protocol

| Excluded fitting group | Frames excluded | Frames used to fit and calibrate | Threshold | Matched / reviewed | Missed |
|---|---:|---:|---:|---:|---:|
| February 24 group | 12 | 30 | .60 | 17 / 179 | 162 |
| March 4 group | 10 | 32 | .50 | 22 / 97 | 75 |
| Original fitting group | 20 | 22 | .45 | 4 / 62 | 58 |

The models matched 43 of 338 sacks when their recording group was excluded, versus 273 when it was included. On the 29 low-count fitting frames, exclusion exposed 102 misses among 120 reviewed sacks, versus 10 when included. Lower-scoring candidates contained a possible one-to-one match for 162 of the 295 total excluded-group misses, including 66 of the 102 low-count misses. This uses labels to measure the opportunity available to a verifier.

We inspected a predetermined sample of 15 missed centers across groups and count ranges. Four showed a clear physical obstruction, two showed no clear obstruction and nine were ambiguous. Examples include an exposed section of red sack behind a truck rail in March 4 frame T04_04 and a white sack partly covered by another sack in T04_07. Glare alone did not qualify as physical occlusion. The sample establishes that such misses exist. It cannot estimate how often occlusion causes a miss. Marked examples in context · All 15 review observations

## One proposed fix

A regularized logistic verifier scores up to 64 candidate centers using pretrained appearance at the point, contrast with its surroundings, proposal score and rank. Its positive targets are one-to-one matches. Lower-scoring candidates that recover a sack missed by the first pass receive double positive weight. Group and class weights balance the fitting objective. At inference the verifier retains every point from the 75-match voting control and adds accepted candidates beyond eight pixels of the existing points.

The primary verifier learned from recording-excluded candidates. A matched verifier used candidates from the same three proposal models, with a cyclic assignment that included the target recording in fitting. Both used identical architecture, optimizer, number of updates, ground truth and calibration rule. Each recording has a different size, so the number of source frames in the corresponding generating fold also varies. Every ranker calibrated on the same 42 recording-excluded training banks. The operating point had to produce zero points on all three training empties. Among eligible thresholds the rule favored group-balanced localization F1 on positive frames. The five development empties had no role in selecting the threshold. Frozen comparison and pass criteria

## Development results

The two reused development groups contain 23 frames and 355 reviewed visible sacks. A match requires a unique reviewed center within eight pixels in the long-side-1024 coordinate system. GAME1 and GAME2 measure spatial count error on 2×2 and 4×4 subdivisions of the foreground ROI. Lower values are better.

| Method | Predicted | Matches | March 12 | Feb 27 | F1@8 | MAE | GAME1 | GAME2 | Low-count matches | Empty false points |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original control | 213 | 71 | 62 | 9 | .250 | 7.48 | 8.70 | 10.17 | 0 | 0 |
| Voting control | 213 | 75 | 68 | 7 | .264 | 7.48 | 8.70 | 10.09 | 0 | 0 |
| Voting plus calibrated raw proposals | 213 | 75 | 68 | 7 | .264 | 7.48 | 8.70 | 10.09 | 0 | 0 |
| Recording-included verifier control | 245 | 88 | 75 | 13 | .293 | 6.35 | 7.74 | 9.57 | 0 | 0 |
| **Recording-excluded verifier** | **401** | **116** | **86** | **30** | **.307** | **5.65** | **7.83** | **9.91** | **0** | **2** |
| Raw score at the primary per-frame budget | 401 | 124 | 93 | 31 | .328 | 5.65 | 8.52 | 10.09 | 0 | 2 |

Relative to voting, the primary gained 18 matches on March 12 and 23 on February 27. Recording-specific F1 rose from .300 to .310 and from .123 to .297, respectively. Overall RMSE fell from 10.01 to 8.03. The model added 188 points for 41 additional matches. False positives rose from 138 to 285, precision fell from 35.2% to 28.9%, and recall rose from 21.1% to 32.7%. March 12 MAE fell from 7.60 to 6.67 and February 27 MAE fell from 7.25 to 3.75.

The three low-count frames contained 2, 2 and 7 sacks and all received zero points. Low-count MAE stayed at 3.67. On all three frames, even label-informed selection among the 64 proposed centers could have produced zero matches at eight pixels. The closest candidate to any reviewed center in the first two frames was more than 250 pixels away in the evaluation coordinate system. In the seven-sack frame the closest distance was 14.3 pixels. The empty-frame false detections occurred once each on February 27 frames D27_02 and D27_03.

At the primary per-frame budget, raw proposal ranking yielded 124 matches versus 116 for the learned verifier. The learned verifier achieved lower GAME1 and GAME2 than this count-matched control. Its regional count totals improved while individual center placement was weaker. The matched-budget control uses the primary model's predicted quota.

## What the evidence supports

Using detections from other recording groups exposes genuine misses that in-sample mining largely hides. The experiment also shows that allocating more points can improve counting on these two recordings. It provides insufficient evidence for a superior point-ranking method, reliable low-count recovery or empty-truck protection. A future attempt to address the low-count cases would first need candidate locations near the reviewed sacks. Threshold changes cannot create a missing location. Full training is unsupported by this gate.

The DGX controller exited successfully after all 1,680 proposal-training updates. An independent scoring implementation checked every development method/frame combination and both kinds of training proposal bank. The result is limited to one initialization, related frames and two previously examined development groups. Per-frame results and gate checks · Independent verification and low-count candidate limits

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 21 files under the folder recording-excluded-proposals/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
