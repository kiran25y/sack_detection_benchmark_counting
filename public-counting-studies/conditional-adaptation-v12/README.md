# Conditional representation audit and one matched adaptation experiment

The owner explicitly authorized finishing V11 first, then a representation audit and one matched source-only adaptation experiment if the fine-resolution candidate fails. This authorization does not resume the paused chat automation. The existing V11 GPU job remains the only active research fit.

## Dependency and evidence gate

The CPU host queue waits for V11's completed state, successful container exit, and the V4 assessment containing both candidate and control with all 1,286 native validation queries and paired image/category intervals. A runtime failure is deferred for diagnosis, never treated as a scientific negative result. The queue verifies the completed checkpoint/prediction hashes before any diagnostic inference.

If the candidate reaches either matched global baseline's MAE, or both image-group and category-component intervals support a candidate MAE improvement over its matched control, the queue defers for review of that gain. This is a conservative routing rule, not a superiority test or automatic approval of a model. Sparse/spatial and category costs still require review. Otherwise, it starts one CPU-only representation audit after the completed comparison. It never starts a GPU fit or automatically invents an adaptation from a negative count result.

## What the code actually learns already

`StructuredCounter` already has three trained `TokenAdapter` modules (projection, channel normalization and a residual pointwise update), shared between query and exemplar features. Thus the earlier suggestion to simply add a small trainable adapter would duplicate an existing capability. The open question is whether its current indirect count/spatial supervision learns useful exemplar discrimination and whether the subsequent matching/decoder preserves that information.

## Prepared audit

Fresh immutable `representation-audit-v12` preserves the 63 original V11 Python files and packages the shared counting dependencies. It adds diagnostics and a conditional queue without changing any active release.

The audit selects 16 source-fitting rows per count stratum (below25,25–149,at least150), using a fixed seed and query order, without selecting high-error images. All 1,066 held selection rows are excluded. Both completed V11 checkpoints are examined on the same 48 rows using one CPU thread and no optimization. Full-refit checkpoints have seen all source-training classes; this is a fitting diagnostic, not held-class accuracy.

For each feature level, compare raw exemplar cosine, learned-adapter cosine, the actual trained matching field and the decoded mass against native point locations. Rank occupied point cells against locations more than two grid cells from an annotated center; ties receive half credit. Report point-cell collisions and unavailable comparisons. Far locations are not semantic negative object labels or segmentation masks. A ranking measure cannot establish missed/merged/duplicate/wrong-object instance counts, native AP or occlusion robustness. CPU FP32 fitting predictions are not replacements for native CUDA BF16 validation results.

Inspect patterns across count strata and the fitting/internal/native error gap. Better raw/adapted ranking with poor decoded localization points toward matching/decoder use of evidence; poor simple cosine alone does not establish that the backbone lacks recoverable information. Matching and cosine are different readouts, so these are localization diagnostics rather than a causal feature test.

## One adaptation decision after the audit

Use the completed audit to specify one intervention and freeze its recipe before native validation. The leading conditional hypothesis is direct, point-derived supervision of the **existing** learned matching representation versus an otherwise identical model trained with the existing objective. It is not yet selected or implemented: if the audit instead localizes the failure downstream, do not add an unsupported feature objective.

The eventual candidate/control must use identical architecture, parameter count, initialization seed, frozen encoder, supplied references, feature grids, source partitions, batch exposure and optimizer configuration, with the intended adaptation difference explicit. Native point maps supply location/count evidence; do not turn arbitrary unlabeled objects into negative class labels. Any coefficient selection uses source-fitting development evidence only and is fixed before native validation. Class-held epoch selection, full-source refitting and atomic model/optimizer/RNG recovery remain required. Different selected refit durations must be disclosed.

Before starting the serial pair, preserve an immutable release and run CPU/CUDA numerical, finite-gradient, native-count-mass and recovery checks. Evaluate MAE/RMSE, sparse/dense strata, class-macro, spatial GAME and paired image/category intervals. No additional seeds or independent-domain/sack scoring until a useful controlled gain is established. Existing native labels/splits, failed attempts, public reserved confirmation and unrelated DGX jobs remain untouched.

Representation matching and supervised exemplar matching have existing precedents, including URM and BMNet; high-resolution multiscale queries are already central to GeCo2. This is a bounded diagnosis and adaptation experiment, not an established novelty claim or a promise of beating the baselines.

## Execution boundary

The host queue can finish the current dependency and run the diagnostic audit unattended. Choosing and executing the adaptation pair still requires inspection of the resulting evidence. The chat heartbeat is currently paused; no automatic chat continuation or adaptation launch is claimed. The user's authorization for that next experiment is recorded here and does not need to be requested again.


October16 update: the owner explicitly resumed the existing heartbeat, now ACTIVE. V11 completed with a supported matched gain (12.4937 vs14.4101) but remains worse than both globals. The preserved queue deferred for gain review; after review, the verified CPU audit was launched once. Earlier paused-status and pending-completion descriptions are historical. See the October16 candidate-result receipt before further work.

October16 audit completion and owner-authorized adaptation: V12 completed96 probes without failure. See V13 protocol for the measured diagnostic limits and the selected single matching-supervision experiment. The plan now fixes both compared refits at56epochs inherited from V11 source-class-held selection, instead of independently selecting unequal durations. The immutable completed V11 checkpoint is the control; no duplicate fit is required after guarded training-equivalence checks. Existing heldouts and native labels remain unchanged.
