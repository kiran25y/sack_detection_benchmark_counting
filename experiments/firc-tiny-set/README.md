# FIRC ordinary tiny-set result — November 6, 2026

**Verified 2026-11-06T12:09:09.135773+09:00: ordinary trainability gate FAIL. The one fit completed exactly 150 updates and 1,200 query presentations. No full training was started. Physical four-state data admission remains unpassed.**

This is a separate ordinary-image implementation check allowed by sections17/20/21 of the supplied plan. The earlier claim that every tiny fitting check needed a physical quartet was too broad; the earlier receipt is retained. This run does not replace or pass the four-state mechanism test.

## Fixed-endpoint results

Four already reviewed fitting images, one seed, official frozen DINOv2 encoder and fresh counting modules. Threshold0.50 was fixed before optimization. The original focal +5point +0.10soft-count loss and all150 updates were retained. No threshold, endpoint or seed selection occurred.

| Image | Truth | Predicted | AE | Point F1@8 | GAME1 | GAME2 | Soft count (diagnostic) | Maximum confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T04_09 | 0 | 0 | 0 | N/A (empty) | 0 | 0 | 0.8505 | 0.0003 |
| 003 | 1 | 0 | 1 | 0.0000 | 1 | 1 | 2.6298 | 0.2751 |
| T04_05 | 21 | 0 | 21 | 0.0000 | 21 | 21 | 21.2683 | 0.0132 |
| T24_10 | 34 | 0 | 34 | 0.0000 | 34 | 34 | 34.2747 | 0.0124 |

Final MAE **14.0000**, RMSE **19.9875**, bias **-14.0000**; pooled F1@8 **0.0000**, positive-image macro F1@8 **0.0000**. GAME1/2 **14.0000/14.0000**. Empty false positives **0**. The pooled matching totals are TP0, FP0, FN56.

Initial-to-final MAE: 14.0000 -> 14.0000, signed difference +0.0000. Pooled F1: 0.0000 -> 0.0000. The initial model is a diagnostic starting point, not a competitive counting baseline.

The frozen ordinary trainability requirements were MAE<=1, pooled and positive-image macro F1>=.80, empty FP=0, and valid finite execution. Failed components: **mae_at_most1, pooled_f1_at_least_point8, positive_macro_f1_at_least_point8**. A correct empty prediction cannot compensate for missed positive sacks. Soft-count values are reported to expose differences between aggregate probability mass and the declared thresholded-point count; they are not substituted as the primary result.

## What the saved outputs establish

The 21- and 34-sack images have soft totals of 21.2683 and 34.2747, but maximum candidate confidences of only 0.01322 and 0.01243. Aggregate probability mass can fit these totals while no individual candidate becomes a confident detection. Across all four images, diagnostic soft-count MAE is 0.75585, whereas the prespecified thresholded-point MAE remains 14. This discrepancy is the observed failure mode; substituting soft counts would conceal the failed point-counting task.

All target points were assigned at every update, native rows and source/reference manifest hashes remained exact, and the original loss arithmetic and learning-rate schedule matched their frozen definitions. Seven additional read-only checks are saved in `artifacts/firc_ordinary_tiny_v1_20261106/supplemental_readonly_verification.json`. Gradient clipping was active on all 150 updates. These observations do not isolate clipping or the soft-count term as a causal explanation: no matched ablation was run in this diagnostic. This result does not justify a longer unchanged recipe, nor does it establish that another pretrained encoder is needed.

## Verification and implementation limits

All 20 independent receipt/coverage checks passed. Controller exited; Docker exited0 with no OOM or guard stop. All150 sequential update records, 1,200 presentations, eight initial/final prediction files, fixed labels and complete final checkpoint hashes were verified. Frozen release/plan hashes match. Matching capacity passed before launch and the worker required full assignment throughout. All logged values are finite.

The worker checked finite trainable parameters/Adam states and equality of frozen encoder contents; the local verifier did not independently deserialize checkpoint tensors. It independently recomputed counts, MAE, RMSE, bias, maximum-cardinality F1@8 and ROI GAME1/2 from saved NumPy arrays. F1 uses the full native-image long-side1024 chart. Empty-image F1 is not credited as positive localization. The diagnostic checkpoint is explicitly barred from research initialization.

Optimization elapsed 14.09 minutes, including endpoint saving/scoring overhead as recorded by the worker. Peak allocated GPU memory 3.85GiB. The two48GiB admissions,32GiB host stop,100GiB disk reserve,24GiB container limit,twoCPU,noextra swap and two-hour maximum were retained. Unrelated jobs were not changed.

Four fitting images and repeated presentations do not establish held-out performance, dense/extreme FSC accuracy, generalization or novelty. This is a bounded test of this initialization, original loss and fixed150-update schedule; failure would not prove that DINO features or all possible point-head recipes cannot learn.

## Physical data result remains separate

The additional search inspected48 new closer-spaced training frames, three contact sheets and four native views around the final unloading. Combined scope is168 distinct archive frames plus12 existing development previews. **Zero physical quartets were confirmed.** Successive removal, moving carriers and visibility changes did not establish the same two edits in all four states. This was not exhaustive review of every raw frame. Five reserved groups remain untouched. No new labels, masks or physical identities were invented.

Full foundation training, the comparative B/C/D screen and the final study remain unstarted. The ordinary result cannot resolve the absent four-state data or establish the claimed mechanism. Any different loss, longer schedule or new diagnostic would be a separate prospective decision, not recovery of this run.

## Saved evidence

- Frozen execution plan
- Analysis: `artifacts/firc_ordinary_tiny_v1_20261106/analysis.json`
- Exact results archive and hashes: `artifacts/firc_ordinary_tiny_v1_20261106/result_outputs.zip`, `collection_receipt.json`
- Point overlay: `artifacts/firc_ordinary_tiny_v1_20261106/figures/endpoint_localization.png`
- Loss trace: `artifacts/firc_ordinary_tiny_v1_20261106/figures/optimization_trace.png`
- Further recording review: `artifacts/firc_recording_search_v3_20261106/admission_review.json`

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 14 files under the folder firc-tiny-set/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
