# FIRC matched count-loss repair result — November 6, 2026

**Verified 2026-11-06T12:48:52.172652+09:00: FAIL.** Exactly 150 additional updates and 1,200 repeated query presentations completed. The original 150-update control was reused, not rerun. Removing the count term did not satisfy the complete tiny-image trainability criteria. Longer training is not launched from this result. The measured gradient conflict is real at initialization, but its removal alone is not a demonstrated complete repair.

## Fixed-endpoint comparison

The only objective change was removal of the 0.10 absolute soft-count Huber term. Pretrained encoder, fresh head/Adam initialization, seed, reviewed images, native references, matching, precision, schedule, clipping and threshold 0.50 were identical. All initial prediction arrays matched the control exactly.

| Image | Truth | Original count | Repair count | ΔAE | Repair F1@8 | ΔGAME1 | ΔGAME2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T04_09 | 0 | 0 | 0 | +0 | N/A | +0 | +0 |
| 003 | 1 | 0 | 1 | -1 | 1.0000 | -1 | -1 |
| T04_05 | 21 | 0 | 0 | +0 | 0.0000 | +0 | +0 |
| T24_10 | 34 | 0 | 0 | +0 | 0.0000 | +0 | +0 |

Signed differences are repair minus original; negative error/GAME differences are improvements. F1 uses maximum point matching within 8 pixels on the native full-image long-side-1024 chart. Empty-image F1 receives no credit.

| Metric | Original | Repair | Signed difference |
|---|---:|---:|---:|
| MAE | 14.000000 | 13.750000 | -0.250000 |
| RMSE | 19.987496 | 19.981241 | -0.006255 |
| Bias | -14.000000 | -13.750000 | +0.250000 |
| Pooled F1@8 | 0.000000 | 0.035088 | +0.035088 |
| Positive macro F1@8 | 0.000000 | 0.333333 | +0.333333 |
| GAME1 | 14.000000 | 13.750000 | -0.250000 |
| GAME2 | 14.000000 | 13.750000 | -0.250000 |
| Empty false positives | 0 | 0 | +0 |

Repair matching totals: TP 1, FP 0, FN 55. The frozen absolute gate was MAE ≤ 1, pooled F1 ≥ 0.80, positive-image macro F1 ≥ 0.80, empty FP = 0, and finite complete execution. Every image also had AE/GAME1/GAME2 protection, with no positive-image F1 regression. Failed components: **mae_at_most1, pooled_f1_at_least_point8, positive_macro_f1_at_least_point8**. All individual protection booleans are retained in `analysis.json`.

## What this says about zero predictions

At the identical initialization, the parameter-gradient norms were focal 5.6772, weighted point 8.7708, and weighted count 96.0480. The count/focal norm ratio was 16.92; their gradient cosine was -0.6613. All four initial soft sums exceeded their true counts, so the count term pushed logits downward; focal pushed matched positives upward. This measured conflict precedes clipping and Adam. It is not by itself proof of the cause of the entire failed trajectory.

The matched endpoint comparison tests whether removing that term is sufficient at this fixed budget. Soft probability sums are descriptive only and never replace thresholded counts. Endpoint probabilities and both zero-update gradient audits remain in the collected output. The prior control's low soft-count error despite zero detections illustrates why sum-of-scores and detected-object counts must be kept separate.

The correct ideal solution can minimize both the instance and count objectives; the observed initial gradient conflict is not a proof that their objectives are mathematically incompatible.

### Saved-array ranking diagnostic after the failed gate

A separate posthoc analysis selected the top N confidence scores with stable candidate-index ties, using the **known true count N**. This is an oracle quota, not a usable counter, a replacement metric, a tuned threshold or a gate rescue. The original control matched only 3 of 56 targets under this quota; the repair matched 44 of 56. By positive image, the repair matched 1/1, 20/21 and 23/34.

For the 21-sack image, all 21 selected candidates were within the fixed 8-pixel tolerance of some target but covered only 20 distinct targets. For the 34-sack image, all 34 selected candidates were near some target but covered only 23 distinct targets. Thus the highest-scoring candidates have much better spatial association, but duplicate coverage and low confidence remain. Both higher-count images have maximum confidence 0.37754, below the frozen 0.50 threshold. The sparse image maximum is 0.50781 and its single detection is correct. This separates learned locations from useful calibrated, one-per-instance output.

This evidence does **not** yet distinguish unstable one-to-one training assignments, insufficient classification convergence, or another cause of the confidence/duplicate problem. No threshold search, NMS, oracle quota deployment, automatic additional arm or full fit was started. The next minimal diagnostic should measure per-object assignment turnover and competing-anchor margins under actual model updates before choosing a new loss or longer training. The earlier IPIC fixed-assignment failure remains relevant history; it cannot simply be relabeled as an untried idea. The current saved endpoints do not contain assignment histories.

## Verification and limits

All 27 independent receipt/coverage checks passed. Controller ended and Docker exited 0, with no OOM or guard stop. The verifier checked all source/input/output hashes, exact control initialization and saved initial arrays, 150 sequential updates, 1,200 presentations, matched LR schedule, actual loss arithmetic, all target assignments, and fixed threshold. The worker checked finite model/Adam states and an unchanged pretrained encoder. The local verifier independently recomputed counts, AE, RMSE, bias, point matching and ROI GAME1/2 from saved arrays; it did not deserialize checkpoint tensors.

Verifier numerical correction: the frozen analyzer first rejected one diagnostic soft sum because Windows and the DGX float32 reduction differed by one float32 unit (0.0000152588). Its original source and failure were preserved. A separately hashed verifier uses a float64 sum reference and a maximum two-float32-ULP discrepancy for this descriptive quantity. Integer counts, predictions, labels, matching, thresholds, all scientific gates and all other checks are unchanged. The repair still fails. See `verifier_numerics_amendment.json` and the original versus corrected analyzer hashes.

Elapsed worker time: 14.16 minutes; peak allocated GPU memory 4.27 GiB. All existing RAM/disk/container/wall guards were retained. No unrelated process was changed. FIRC cumulative optimizer calls are 307: 7 prior engineering calls, 150 original tiny updates, and 150 repair updates. Both new gradient audits made zero optimizer calls.

These are four repeatedly fitted images in three recording groups, one seed, and no held-out evaluation. Counts 21 and 34 are higher within this sack subset, not FSC extreme counts. Neither these results nor a correct empty image establish generalization. The diagnostic checkpoint remains ineligible as research initialization. Zero valid physical four-state quartets are admitted; no physical identity or masks were invented. Five reserved recording groups remain untouched.

This is an objective ablation, not novelty. Point classification plus point regression is established, including the [official P2PNet criterion](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/blob/main/models/p2pnet.py); that precedent does not imply its exact loss or architecture is identical to ours. Earlier IPIC count-loss removal is acknowledged rather than presented as an untried idea.

## Evidence

- Frozen recipe: `artifacts/firc_count_ablation_recipe_v1_20261106/plan.json`
- Verified analysis and all component checks: `artifacts/firc_count_ablation_v1_20261106/analysis.json`
- Collected outputs and archive receipt: `artifacts/firc_count_ablation_v1_20261106/collected/`, `result_outputs.zip`, `collection_receipt.json`
- Reviewed overlay and optimization trace: `artifacts/firc_count_ablation_v1_20261106/figures/`
- Original control remains in `artifacts/firc_ordinary_tiny_v1_20261106/`.

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 16 files under the folder firc-count-loss/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
