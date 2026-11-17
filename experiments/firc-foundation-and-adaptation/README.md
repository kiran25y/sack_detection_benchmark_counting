**November 8 verified endpoint.** All 3,800 ordinary training updates completed. Development MAE was 11.8261 at the training-selected threshold of 0.35, with pooled point F1 of 0.1142 and 25 of 355 sacks localized. Practical readiness failed. Endpoint report

# FIRC architecture: ordinary public training and sack adaptation

**Actual training launched November 6 at 15:21 KST.** Attempt `firc_ordinary_training_v1_20261106`, controller PID 1195705, container `firc-ordinary-training-v1-20261106`. At 15:26 KST: 25 durable updates and 47 logged updates; no OOM, worker failure or guard stop. The initial training head exactly matches the fresh preflight initialization, not a fitted diagnostic checkpoint.

The owner explicitly instructed broader training after receiving the failed calibration result. That authorizes this new ordinary-supervision run. All previous failed accuracy gates remain failed; this is not a retroactive pass or a restart of an old experiment. The four-state intervention study and nine-arm matrix are not launched because they lack the required physical data.

## Frozen run

| Stage | Updates | Query presentations | Source proportions | Initialization |
|---|---:|---:|---|---|
| Public foundation | 800 | 6,400 | 3,200 FSC-147 + 3,200 D2S | Official frozen DINOv2; fresh counting modules; detail disabled/frozen |
| Ordinary adaptation | 3,000 | 24,000 | 12,000 sacks + 7,200 D2S + 4,800 FSC-147 | Shared foundation; detail enabled from its fresh alpha-zero state; Adam/scheduler reset |

Seed 1729. Eight independently referenced queries are accumulated per optimizer update. The current model pools reference prototypes across a microbatch, so ordinary queries use microbatch one and accumulation eight; unrelated categories cannot accidentally share one reference prototype. This preserves eight query presentations and their mean loss, while differing numerically from a four-query microbatch.

The approved sack cohort contributes **all 42 training frames, 338 points and three recording groups**. Neither the 23 development frames nor the five reserved groups enter optimization. Source sampling is frozen before training: FSC category then image, D2S target category then image, sacks recording group then frame. D2S predicts only the chosen category with valid same-category exemplars; the old unconditioned all-products IPIC mode is not reused.

The source manifest contains 8,033 eligible public images and 42 sack training images. The sampled foundation uses 3,664 distinct rows; adaptation uses 5,183 distinct rows. Repeated exposures are not independent images or scenes.

## Recipe and amendments

Frozen official DINOv2 ViT-B/14 register checkpoint; counting neck, conditioning and point head learn from fresh initialization. AdamW peak learning rate 0.0002, betas 0.9/0.999, epsilon 1e-8, weight decay 0.05 except biases/one-dimensional normalization parameters, gradient clipping 1.0. Foundation warms up for 80 updates; adaptation warms up for 150; each uses cosine decay to 1% of peak. The exact rate tables are saved in the plan.

Loss is focal classification plus five times the point regression loss. The previously diagnosed antagonistic 0.10 soft-count term stays removed. No interaction, mixed-response or mask loss is used. These are standard components and do not establish a new contribution.

Augmentation preserves all supervised content: full public image or approved sack ROI; long side 672/784/896; horizontal flip probability 0.5; two of brightness/contrast/saturation in 0.8–1.2. No arbitrary crop, erasure, generated removal or fabricated quartet. Reference crops remain native training exemplars, aspect-preserved into 224-square canvases. Query padding follows the declared multiple of 112.

Checkpoints save every 25 updates and at fixed stage endpoints. No intermediate development scoring, best-checkpoint selection, extra seed, automatic retry or extra update budget. The final adaptation checkpoint is the designated evaluation model.

## Checks completed

- 13,605 distinct prescribed target/resize/flip configurations passed complete reachable-anchor matching. Maximum native-coordinate round-trip error: 2.27e-13 pixels. Existing native image audits and approved labels are reused.
- Six disposable real-data CUDA updates covered foundation and adaptation graphs, all three resolutions, sparse and empty sacks, public exemplars, and the 1,912-target FSC image.
- Both phases reproduced the next update exactly after restoring nonempty Adam/model/RNG states: maximum observed model and optimizer differences zero. Preflight weights were discarded.
- Every target was matched, all gradients/model/optimizer states remained finite, and frozen DINO weights stayed unchanged.
- Preflight median 7.40 seconds/update, including the deliberately extreme FSC query; peak allocated GPU memory about 1.24 GiB. The first 47 ordinary training updates had a recent median of 5.59 seconds. This is an early rate, not a completion guarantee.

All existing memory/disk boundaries remain: two 48 GiB host-RAM admission checks, 32 GiB host stop, 100 GiB disk reserve, 24 GiB container and equal swap limit, two CPUs, no network and read-only inputs. The newly declared full-run wall cap is 12 hours. No unrelated process is modified.

## Completion and evaluation

Verify exactly 800 + 3,000 durable updates and 30,400 sequential query records, frozen plan/data/code identities, native target semantics, finite worker checks and final checkpoint hashes, with successful controller/Docker exit and no OOM/guard stop. The local verifier hashes checkpoints without independently deserializing tensors.

Then run exactly **65 inference-only queries** on the fixed endpoint: all 42 training frames first; choose one global threshold from 0.10–0.90 in steps of 0.05 by training MAE, ties toward the higher threshold; commit it before the 23 development queries. Retain fixed 0.50 as a full secondary result. Report MAE/RMSE/bias, one-to-one point F1, GAME1/2, low-count, empty-image and per-recording-group effects. No reserved model scoring is queued.

The frozen practical readiness report requires development MAE ≤7.7174, pooled and positive-image macro F1@8 ≥0.50, TP on at least 14/18 positive frames, training positive-image macro F1 ≥0.80, low-count MAE ≤2 and F1 ≥0.50, and zero empty false positives. A completed fit or a small MAE alone is not a pass. This readiness report is not a CVPR acceptance criterion.

Final deployment evidence still requires reviewed exhaustive labels on the reserved cohort and a frozen final evaluation. Its 51 existing frames in five conservative recording groups are not verified independent physical episodes/sites. A new contact-sheet pre-screen is recorded separately, with no invented counts. Historical V11/V14 narrow-lane exposure must be audited per baseline. No final accuracy or paper acceptance can be promised before measurement.

D2S grapes/kiwis/apples overlap FSC held-out categories: this is joint-data training, not strict unseen-category transfer. This single ordinary model is a functioning-baseline attempt; it does not test the proposed factorial mechanism or replace its matched controls.

## Reproducible evidence

- `study_protocols/firc_ordinary_training_prepared_20261106.json`
- `study_protocols/firc_ordinary_cuda_verified_20261106.json`
- `study_protocols/firc_ordinary_training_v1_20261106_launch.json`
- `artifacts/firc_ordinary_training_recipe_v1_20261106/plan.json`
- `scripts/inspect_firc_ordinary_training.py --save`
- `scripts/verify_firc_ordinary_training_completion.py`
- `scripts/prepare_firc_ordinary_evaluation_once.py`
- `scripts/analyze_firc_ordinary_evaluation.py`

Plan SHA256: `9a36b200215d6e75877897ced99f0c581e2e720ed941c6e10773e0f8f15c82dd`.

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 9 files under the folder firc-foundation-and-adaptation/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
