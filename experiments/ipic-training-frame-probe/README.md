# IPIC: why the current result is insufficient and what to test next

Verified November 4, 2026, 14:06 KST. The six-arm result is preserved. This follow-up audits saved outputs and adds only inference on existing training images; no optimization, checkpoint selection, new threshold search or full training occurred.

## The decisive finding is a generalization gap

The primary B0 control, A, nearly memorizes its 20 training frames but fails to transfer to the separate development recording. At the same fixed 0.50 threshold:

| Saved model | Training MAE, 20 frames | Training F1@8 | Development MAE, 11 frames | Positive-development F1@8 |
|---|---:|---:|---:|---:|
| Public foundation | 3.10 | 0.0000 | 22.1818 | 0.0000 |
| A: focal + point | **0.05** | **0.9929** | 22.1818 | 0.0000 |
| B: global + local | 0.75 | 0.9072 | 22.0000 | 0.0000 |
| E: global only | 0.90 | 0.8762 | 22.0000 | 0.0000 |

A finds 61 of the 62 training points with no false positives at this tolerance, missing one point in frame 006. These are seen-data fitting results, not a publishable accuracy result. The earlier four-image FSC trainability check was not evidence of sack transfer. This check now establishes that the implemented base objective can fit the sack labels. It does not identify one exclusive cause of the transfer failure: recording appearance, count distribution, object scale, background dependence, reference conditioning and limited diversity remain intertwined.

The inference probe used the verified foundation and final A/B/E checkpoints selected by their prescribed B0 roles, not because one won the development table. It completed **80 query forwards and four reference-batch forwards**, with zero optimizer calls. All four model-state hashes were unchanged before/after inference. Docker/controller exited successfully with no OOM/guard stop; all checkpoint/source/data identities and collected output hashes matched. Minimum sampled available host memory was 76.39 GiB. Metrics were recomputed locally from saved predictions with the independent point matcher. No new development inference was needed.

## What the implementation and data audit found

**Losses are already implemented and active.** All 48,000 logged pair losses reconstruct as the prescribed focal + 5×point objective, plus global 0.10 and local 0.25 only for their assigned arms, within FP32 rounding tolerance. The absolute soft-count term is zero by the approved amendment; it was not accidentally omitted. This audit found no coefficient or native-to-network-to-native coordinate error in the reviewed code path. It is not a proof against every possible implementation issue.

**The natural sack distribution is extremely narrow.** Training has 20 frames, 62 points, one recording group, counts 1–11 and no natural empty frame. Development has 11 frames, 244 points, one different recording group, counts 0–34. All ten nonempty development frames exceed the maximum training count. These are a combined camera/appearance/count-distribution shift, not an isolated dense-count test. Neither group supplies ≥150/≥900-count sack evidence.

**Augmentation is already substantial.** Each arm saw 8,000 natural sack views: 4,043 flipped, 2,396 affinely transformed, and long sides 768/896/1024 used 2,722/2,700/2,578 times. Appearance changes included contrast, brightness, saturation, gamma, noise, blur and JPEG; 3,229 views received two operations. There were another 4,000 public-data pairs per arm, including 1,000 D2S composites. Repeated views do not create independent recordings. Arbitrarily increasing blur, noise or occlusion could remove useful visual evidence.

**The proposed local mechanism has no direct natural sack mask supervision.** All 4,000 sack pairs per arm are point-only; their local term is zero. The local term is learned from D2S masks/composites and expected to transfer. That protects uncertain sack boundaries, but it limits what this experiment tests: it cannot establish a local physical sack-intervention mechanism. B and D each worsen final development-selected MAE by 0.8182 against matched global-only controls E/F.

**Low confidence is not the whole problem.** In a posthoc diagnostic, select the top N points where N is the true image count. This uses ground truth and is not deployable. Even with the number of predictions forced correct, only 22–34 of 244 points match truth within 8 pixels at normalized 1024 scale: pooled F1 0.090–0.139 across A–F. Correcting the scalar count does not recover the locations. The original final checkpoints and threshold grid remain unchanged.

**The overlays show substantive recognition errors.** In the F control, the empty truck receives seven detections at the prescribed development-selected 0.10 threshold, including lettering and worker regions. Nonempty frames contain missed sacks and false positives. These are visual observations, not proof that one particular feature causes the errors. See the complete F overlay and B overlay. Both contain all eleven frames.

**Loss scale merits diagnosis, not an automatic clipping change.** Every update in every arm clipped the global gradient norm. Median pre-clip norms were about 4.01/4.09 for A/C and 9.53–9.84 for the pairwise arms. On FSC in the first 100 updates, weighted global-loss values were about 2.98 versus base values about 1.11. Loss values and global norms do not reveal each objective's gradient direction or its post-Adam contribution. Clipping alone is not evidence of a bug or justification for weakening the guard.

The sum of focal sigmoid scores is also not an independently calibrated count. For A, clean training hard-count MAE is 0.05 while mean soft mass is 25.79 against mean truth 3.10. Many low-confidence background proposals contribute to that sum. Adding a soft-mass penalty without checking its interaction with object classification would constrain a different quantity from the reported thresholded point count.

## Next bounded experiment: improve transfer while keeping the functioning loss fixed

The immediate priority is data coverage and transfer, not another invented loss. Existing source recordings can supply the required images; new physical acquisition is not assumed. Do not move the current development frames into training or relabel their results as independent testing.

1. Prepare additional, independently grouped training recordings with reviewed empty/low-count trucks and visibly stacked loads covering the development count range and appearance variation. Include verified non-target clothing, lettering and folded/empty bags in the reviewed scope. Do not infer background labels from distance to a point or use uncertain masks as cutouts. Preserve current approved labels.
2. Freeze a development panel spanning at least two recording groups and a separately untouched final-test group before model fitting. The already-used development recording remains development. Confirm recording separation and exact image duplicates before splitting. Group count is a pilot minimum, not a guarantee of population-level evidence.
3. Run one **2×2, matched 200-update-per-arm transfer pilot**, only when these data are ready. All four arms start from the same verified public B0 foundation, with reset Adam, the same schedule/clipping, two reference crops, input scope, batch size and 800-pair source budget per 100 updates. Hold focal + 5×point fixed; do not add global/local soft-mass losses. The two factors are recording coverage and augmentation. The four cells are original/broadened training coverage crossed with current/conservative augmentation. Keep 50% sack, 25% FSC and 25% D2S sampling and identical public draws; sack-recording choice is the intended data intervention. Match all random seeds and pair draws wherever applicable.

For the conservative augmentation factor, retain the current whole-image resize/flip geometry, but remove added blur, noise and JPEG from natural sack views; keep only bounded brightness/contrast/saturation/gamma at the existing ranges and at most two operations. Generate its appearance choices from separate fixed seeds, so removing operators does not silently change geometry or sample order. All four cells retain original labels and reference inputs. This tests one augmentation package; it does not isolate each appearance operator. Public augmentation is identical across cells.

Freeze exact image lists, source hashes and the pilot protocol before launch. Compare the broadened-data/conservative-augmentation cell against all simpler cells; do not select a different favorable cell after seeing results. The proposed pilot gate is ≥10% lower overall development MAE and ≥0.05 higher F1@8, with RMSE/GAME1/GAME2 and the prespecified low-count stratum protected within max(0.1, 1%) of each corresponding control; empty-frame false positives must not increase. Report every recording group and low/high-count subgroup, including failures. Threshold 0.50 is primary; 0.10 is fixed secondary diagnostic, not a new calibration search. Do not call 18–34 objects an FSC dense/extreme result. A 200-update screen is a progression decision, not evidence of convergence or final accuracy.

This pilot is **prepared conceptually, not launched**: the expanded reviewed groups and frozen evaluation panel do not yet exist. The exact run manifest, inference guard and implementation tests still depend on those data. At the measured A–F throughput, 800 total pilot updates would take roughly 2–4 hours after data preparation and resource admission; annotation/provenance review time is additional and not yet reliably estimable. If transfer remains weak, stop extending this custom stack rather than spend another full run on unverified losses.

## What remains necessary for a paper

Current evidence is insufficient for a defensible main-track method claim. We need a method that outperforms matched controls on independent recordings, a justified counting mechanism beyond standard point supervision, comparable pretrained counting/detection baselines under the same labels/scope/reference inputs, and repeatability across seeds after the pilot. The current sack MAEs cannot be compared numerically with the older FSC V11/CountGD++/GeCo2 MAEs: they use different datasets and query protocols.

Focal loss already addresses foreground/background imbalance; using it or a routine variant is not itself a new counting contribution. [Lin et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html). One-to-one point supervision and evaluating localization alongside count error also have direct precedent in [P2PNet, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Song_Rethinking_Counting_and_Localization_in_Crowds_A_Purely_Point-Based_Framework_ICCV_2021_paper.html). The proposed transfer pilot tests whether the data/recognition problem is tractable; it is not a novelty claim or an acceptance promise. The IPIC local-loss contribution remains unsupported and cannot be rescued merely by obtaining a better thresholded total.

Evidence: `artifacts/ipic_sack_failure_audit_20261104/analysis.json`, `artifacts/ipic_sack_train_fit_probe_v1_20261104/analysis.json`, and `study_protocols/ipic_sack_failure_audit_20261104.json`. The original A–F completion receipt is unchanged.

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 9 files under the folder ipic-training-frame-probe/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
