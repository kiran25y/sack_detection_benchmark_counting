# Pretrained sack references — verified November4,2026

**Completed and independently verified at15:18KST:** all58 prescribed counter-prediction calls, all23 native development frames per system, zero optimizer/persistent updates, four model containers/controller exit0, no OOM or guard stop. Hooks recorded114 backbone forward invocations, including separate query/reference encodings and compatibility calls;58 is not an equal-backbone-compute claim. All231 current source/input/weight identity checks and native identity/repeat checks passed. Original A–F predictions were reused and rescored locally; no A–F model ran again.

Only the original11 owner-reviewed frames are scored below:244 filled sacks in one dependent recording group. The12 additional development predictions are preserved but unscored pending complete labels. No new test, population MAE, source FSC score, sparse1–11 protection or dense/extreme claim is available from this subset.

| System/settings | MAE | RMSE | Bias | GAME1 | GAME2 | Pooled F1@8 | Empty-frame FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPIC control A, primary0.50 | 22.1818 | 23.9203 | -22.1818 | 22.1818 | 22.1818 | 0.0000 | 0 |
| IPIC control A, secondary0.10 | 10.8182 | 13.8269 | -0.8182 | 16.0909 | 23.7273 | 0.0960 | 17 |
| CountGD++, author0.23 external | 15.8182 | 18.6548 | +7.6364 | 24.9091 | 29.4545 | 0.1818 | 9 |
| GeCo2 external adapter, native postprocess | 18.8182 | 20.6904 | -18.8182 | 19.7273 | 20.2727 | 0.0996 | 0 |

Pooled F1 combines point matches across images, normalized to a1024-pixel long side with an8-pixel radius. It differs from the mean per-image F1 in previous tables; the complete analysis retains both. One empty image inflates macro F1 when it is correctly empty, so positive-image and pooled metrics are essential. FP/FN under this metric include localization displacement relative to the reviewed point convention; they are not independently verified semantic instance labels.

CountGD++ reduces MAE relative to the primary0.50 IPIC control but worsens GAME1/2 and empty-image false positives. At0.10, IPIC control A has lower MAE/GAME than these references but pooled F1 is only0.0960. CountGD++ matches52 of244 reviewed locations with276 unmatched detections; GeCo2 adapter matches14 with23 unmatched detections and230 missed points. No method has a defensible overall count/localization advantage here. No threshold or candidate was selected after seeing these results.

## Reused six-arm results at the fixed secondary0.10 threshold

| Old arm | MAE | GAME2 | Pooled F1@8 | Empty-frame FP |
|---|---:|---:|---:|---:|
| A | 10.8182 | 23.7273 | 0.0960 | 17 |
| B | 11.3636 | 22.8182 | 0.1195 | 14 |
| C | 15.6364 | 30.3636 | 0.1115 | 11 |
| D | 11.1818 | 21.1818 | 0.1162 | 12 |
| E | 10.5455 | 21.0909 | 0.1224 | 10 |
| F | 10.3636 | 22.0000 | 0.1362 | 7 |

These are saved update1000 outputs, not newly trained baselines. C's earlier development-selected threshold was0.15; the table here deliberately uses the same fixed secondary0.10 for all arms. F is not promoted because its MAE happens to be smallest. The prior local-versus-global contrasts remain unchanged.

## Input and comparison limits

All methods receive the full native image and the same two approved external training-reference crop pixels; predicted centers are restricted to the same predeclared coarse camera ROI. The new baselines use no truth points, counts, per-instance masks, negative prompts or inference threshold tuning. The old640×640 banks with three in-image prompts cannot replace these inputs, as verified by the reuse audit.

CountGD++ uses the unchanged author external-reference path with two unresized crops packed on a neutral canvas, its author0.23 threshold and global800/1333 preprocessing. It does not use enhanced crops or SAM test-time normalization. GeCo2 requires a disclosed external-reference adapter: the atlas is separately encoded at the query scale and supplies its three native ROI feature levels. Its same-image identity check exactly/numerically passes the frozen1e-5/1e-6 tolerances, but external feature context and reference packing remain an adaptation. **This is not an official native external-reference GeCo2 result.** Both references have different pretraining, capacity and inference costs from IPIC, and neither receives the target optimization that IPIC received. This is a pretrained-system comparison, not a matched-capacity causal comparison.

Fixed witnesses dev_00/dev_01/dev_06/dev_10 show background/conveyor detections, missed stack locations and displaced centers. The training crops are pale sacks, whereas these development frames contain red printed sacks. That visible input difference is consistent with transfer difficulty but does not isolate its cause. Do not change reference selection or the foreground ROI after viewing errors.

View the fixed witness figure. Green rings are reviewed points; red crosses are predicted centers. Figure crops are for display only; model inputs were full images.

## Decision

Keep the single prepared recording-coverage × conservative-appearance pilot with all original losses/controls and its fixed gate. It is still unlaunched because20 new training and5 new development frames require exhaustive review. Do not add another loss, contrastive model, ensemble or repeat the completed six-arm screen. The current evidence supports a transfer/localization problem, not an established novel mechanism or paper-ready result. A passing pilot would justify further investigation, not a publication claim.

Evidence: completion receipt, complete numerical analysis, frozen protocol. A container-owned checkpoint caused an initial host-user hash-read denial; its receipt is retained. A CPU-only read-only hash container completed verification without changing permissions, loading a model or repeating inference.

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 30 files under the folder pretrained-counting-references/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
