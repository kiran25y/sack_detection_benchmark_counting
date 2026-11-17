# FIRC matched learning-rate continuation: verified result — November 6, 2026

**Both arms completed 150/150 new updates. Primary comparative gate: FAIL.**

Verified at 2026-11-06T13:41:44.406935+09:00. Controller completed/exited and Docker exited 0 without OOM or guard stop. All 32 independent receipt/coverage checks passed. Exactly 300 new optimizer calls and 2,400 query presentations were saved. Each arm ends at cumulative diagnostic step 300 after independently restoring the same step-150 model, nonempty Adam state and RNG.

| Fixed 0.50 inference, four fitting images | Reference LR 0.0002 | Primary lower LR 0.00002 | Lower minus reference |
|---|---:|---:|---:|
| MAE | 9.5000 | 13.7500 | +4.2500 |
| RMSE | 15.5242 | 19.9812 | +4.4571 |
| Bias | -9.5000 | -13.7500 | -4.2500 |
| Pooled F1@8 | 0.4865 | 0.0351 | -0.4514 |
| Positive-image macro F1@8 | 0.6584 | 0.3333 | -0.3251 |
| GAME1 | 9.5000 | 13.7500 | +4.2500 |
| GAME2 | 9.5000 | 13.7500 | +4.2500 |
| Empty false positives | 0.0000 | 0.0000 | +0.0000 |

| Image | Truth | Reference count / F1@8 / GAME1,2 | Lower count / F1@8 / GAME1,2 |
|---|---:|---|---|
| T04_09 | 0 | 0 / N/A (empty) / 0,0 | 0 / N/A (empty) / 0,0 |
| 003 | 1 | 1 / 1.0000 / 0,0 | 1 / 1.0000 / 0,0 |
| T04_05 | 21 | 13 / 0.7647 / 8,8 | 0 / 0.0000 / 21,21 |
| T24_10 | 34 | 4 / 0.2105 / 30,30 | 0 / 0.0000 / 34,34 |

## Frozen decision

The primary comparison requires MAE improvement ≥1, positive macro F1 gain ≥0.10, per-image AE/GAME1/GAME2 and single-sack F1 protection, and primary absolute readiness. Absolute readiness requires MAE ≤1, pooled and positive macro F1 ≥0.80, exact single-sack count, zero empty false positives and complete engineering checks.

- reference_lr absolute readiness: FAIL. Failed components: mae, pooled_f1, positive_macro_f1.
- lower_lr absolute readiness: FAIL. Failed components: mae, pooled_f1, positive_macro_f1.
- Primary comparison: FAIL. Failed components: mae_gain, positive_f1_gain, all_protections, primary_absolute.
- Failed per-image protections: T04_05_ae, T04_05_game1, T04_05_game2, T24_10_ae, T24_10_game1, T24_10_game2.

Predeclared ordinary-pilot candidate: **none**. Neither arm is promoted to longer training.

## Interpretation and prior evidence

The completed assignment probe did not support frequent switching: no switches for 21 targets and one one-time switch for 34 targets. Its negative criterion prevented an unmotivated fixed-assignment fit. The LR comparison tests whether a tenfold reduction repairs confidence while preserving localization. It does not establish that update size is the unique cause, or that this architecture can never learn with another budget. The first 12 reference updates deliberately replay the disposable probe; they are not independent replications.

The original count-loss repair ended at counts 0/1/0/0 for truth 0/1/21/34. Any nonzero endpoint here is a measured change, but a count is not sufficient evidence of correct object localization. The fixed matching/F1/GAME results above determine the conclusion. `posthoc_confidence_separation.json` uses labels and assignment identities only for descriptive diagnosis; it does not alter the threshold, counts, gate or endpoint selection.

The saved endpoint diagnosis narrows the remaining issue. Both arms have one assigned point within eight pixels (long-side-1024 chart) of every one of the 56 fitting targets. For the reference arm, the least-confident assigned target scores 0.3702254 and the most-confident unassigned candidate across all four images scores 0.3276683: a positive score gap of 0.0425571. Thus the assigned true and background/duplicate candidates are separable on these fitting outputs, while the fixed 0.50 threshold rejects many true targets. The lower-rate arm has a negative global gap of -0.1022332. Assignment identities here use truth; this is descriptive fitting evidence, not deployable or held-out accuracy. No replacement threshold was chosen, no threshold candidates were scored, and the failed 0.50 gate remains failed.

The next bounded question is whether one training-only global confidence calibration transfers to the already-reviewed development groups while preserving localization and empty/low-count behavior. This should be an inference-only test with fixed-0.50 controls, both saved arms, unchanged pixels/references/ROI and no extra optimizer calls. Freeze the calibration rule before seeing any new development outputs. This is standard calibration engineering, not novelty, and cannot authorize the physical four-state method. The prepared design is `study_protocols/firc_calibration_check_proposal_20261106.json`; it is not launched.

No earlier IPIC, CounTR or failed FIRC fit was restarted. FIRC work now totals 620 optimizer calls: seven engineering calls, 150 original tiny updates, 150 no-count repair updates, 13 disposable assignment-probe calls and this 300-call comparison. These separately recorded, owner-authorized repairs exceed the initial proposal’s 200-call diagnostic planning allowance; they must not be reported as occurring within that original allowance. The first assignment-probe version failed before any optimizer call and remains preserved.

## Verification and limits

The frozen analyzer checked release/data/plan identities, exact saved initial arrays, authentic initial model/Adam identity, all 300 sequential logs, loss arithmetic, LR schedules, dynamic assignment use, matching-call budgets, unchanged encoder and final checkpoint hashes. The worker checked finite model/Adam tensors; the local verifier did not deserialize checkpoint tensors. Independent NumPy arithmetic recomputed counts, MAE, RMSE, bias, maximum point matching and ROI GAME1/2 from saved arrays. No model inference was added for verification.

The original frozen Windows analyzer flagged exact LR equality at three steps per arm: its math-library cosine rounding differs from the DGX by at most 2.710505431213761e-20 in LR. Its original source and `analysis.json` are preserved. A separate read-only DGX stdlib calculation exactly reproduced all 300 logged rates; the saved log/plan hashes match the local archive. `analysis_verified.json` reconciles only these schedule checks using that exact table. No tolerance, optimizer, metric, threshold, scientific gate or negative decision changed. See `verifier_schedule_reconciliation.json`.

Worker elapsed time: 28.26 minutes. Guards were unchanged. The two diagnostic checkpoints remain ineligible as research initialization.

These are four repeatedly fitted images, one seed and three recording groups, with no development or reserved scoring. Counts 21 and 34 are higher within this sack subset; they are not FSC extreme-count evidence. LR changes are established optimization practice, not a novel counting contribution.

## Full training and final deployable accuracy

Full FIRC method training has not started. Zero physical four-state quartets are admitted. The public-data reachability preparation passed 24,099 checks on 8,033 source-training images, but a correct category/reference public loader, augmentation ledger and mixed-public CUDA check are separate prerequisites. The old IPIC all-products loader must not be used unchanged.

Final deployable accuracy is **not established**. The reserved inventory contains 51 native images in five conservative recording groups, with zero approved exhaustive labels and zero new scored predictions. Those groups are not verified independent physical episodes or sites. The 65 approved fitting/development frames remain unchanged. Final scoring needs reviewed labels/visibility, a frozen inference recipe and checkpoint, and fair same-cohort references. Ground-truth quotas, per-image threshold changes, fitted-image metrics and reused development scores cannot supply that final accuracy.

## Evidence

- `artifacts/firc_lr_continuation_v1_20261106/analysis_verified.json` — all metrics and reconciled checks; original `analysis.json` remains preserved.
- `artifacts/firc_lr_continuation_v1_20261106/collected/` and `result_outputs.zip` — preserved raw outputs.
- `artifacts/firc_lr_continuation_v1_20261106/figures/` — endpoint overlays and complete trajectories.
- Diagnosis and data-readiness checks.
- `artifacts/firc_final_evaluation_readiness_v2_20261106/pool_and_contract.json` — reserved pool and inference contract.

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 25 files under the folder firc-learning-rate/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
