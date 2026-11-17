**November 6 verified FIRC LR comparison: FAIL.** Both arms completed150/150newupdates. Reference MAE9.5000; lower-LR MAE13.7500. Absolute readiness: reference False, lower False. Fitting-only evidence; final deployable accuracy remains unestablished. Verified result. Historical receipts remain below.

# FIRC confidence and learning-rate diagnosis — November 6, 2026

**Current state: the matched two-arm learning-rate continuation is running. No final result or longer-training approval has been established.** Reference peak LR is 0.0002; primary lower peak LR is 0.00002. Each arm independently restores the same authentic no-count step-150 model, Adam state and RNG, then takes exactly 150 new updates. The frozen endpoint is used; intermediate favorable outputs cannot replace it.

## What the completed diagnostic established

The preceding no-count repair learned some useful locations but failed ordinary fitting readiness: counts 0/1/0/0 for truth 0/1/21/34, MAE 13.75 and pooled F1@8 0.0351 at the fixed 0.50 threshold. Ground-truth-quota ranking was descriptive only and is not a deployable result. See the preserved repair report.

The assignment probe completed 13 disposable Adam calls, including one recovery reference and 12 trajectory updates, with zero persistent updates or research checkpoint. All 13 engineering/coverage checks passed, the controller ended, and Docker exited 0 without OOM or guard stop. Exact initial arrays, loss/output-gradient parity and authentic model/Adam recovery passed.

Assignment switching did **not** meet the prespecified criterion. No target switched on the 21-sack image; only one of 34 targets switched once on the higher-count image. No target switched twice. The required fraction switching at least twice was at least 10% in either image. A fixed-assignment training comparison is therefore not launched. This also avoids repeating the earlier negative IPIC fixed-assignment mechanism screen.

Confidence nevertheless oscillated during the LR restart. At updates 8/10/12, hard counts on the 21-sack image were 9/0/22, and on the 34-sack image 7/0/15. These observations motivate testing update size; they do not prove that LR is the only cause or establish generalization.

The first probe attempt failed before any optimizer update because its recovery comparison mixed CPU and CUDA tensors. Its failure outputs remain preserved. Version 2 compares corresponding CPU copies without changing tolerances or the scientific protocol.

## Current matched continuation

- Attempt: `firc_lr_continuation_v1_20261106`; launched November 6 at 13:05:49 KST.
- Arms: `reference_lr` and primary `lower_lr`; 150 new updates each, 300 total calls and 2,400 query presentations.
- Identical starting model **and** nonempty Adam state, seed 1729, four fitting images, reference crops, dynamic matching, focal plus five-times point loss, clipping, resolution, BF16 runtime and no augmentation.
- Identical 15-step warmup/cosine shape; only the LR multiplier differs by tenfold. Threshold remains 0.50. The first 12 reference updates deliberately replay the disposable probe and are dependent observations.
- Existing RAM, disk, container and wall-time guards remain unchanged. No automatic retry, budget extension, threshold search, new arm or full fit.

Each arm's absolute readiness requires MAE ≤1, pooled F1@8 ≥0.80, positive-image macro F1@8 ≥0.80, no empty false positives and exact single-sack count. The primary comparative gate also requires MAE improvement ≥1, positive macro F1 gain ≥0.10 and no per-image AE/GAME1/GAME2 or single-sack F1 regression. All engineering checks must pass.

The predeclared decision permits only a broader ordinary-image pilot if an arm passes absolute readiness; the primary also needs all per-image protections. If neither passes, longer training does not start. These are standard optimization checks, not novelty. Diagnostic checkpoints remain barred as research initialization.

## Final paper evaluation is a separate step

The reserved pool now contains 51 native images across all five still-reserved recording groups. File and decoded-pixel checks find no exact overlap with the 65 reviewed fitting/development frames. No new model predictions have been scored on this pool, and no exhaustive labels in it have yet been approved. The existing snapshot selection is not a population sample; recording groups are not verified independent physical episodes or sites. Historical model exposure must be audited separately for each baseline.

The deployment contract accepts an image, two fixed training reference crops and a predeclared camera/task ROI. It forbids ground-truth counts, points, masks, oracle quotas and per-image threshold tuning. Final metrics require reviewed visibility/labels, a frozen checkpoint and inference recipe, and fair same-cohort baselines. Reused development metrics or tiny-set fitting results cannot serve as final deployable accuracy.

Zero physical four-state quartets are admitted. An ordinary-image learning result cannot clear that separate data requirement or establish the proposed mixed-response contribution.

## Checks needed before a broader ordinary foundation

The previous IPIC public loader cannot be reused unchanged. Its D2S `all_products` mode deliberately supplies no visual references, whereas this FIRC model requires one to three positive reference crops and predicts the queried category. The FIRC proposal explicitly requires target-category-only D2S labels. Also, the current FIRC forward shares its reference prototype across the query batch: mixing different categories with one common reference set would change the task. A public runner must use correctly associated reference inputs per query or a genuinely common-category bundle.

An eligible subsequent ordinary foundation therefore needs fresh official DINO initialization and fresh common modules, the accepted amended objective, correctly paired category/reference/point semantics, full-image geometry and augmentation ledgers, matching-capacity checks at all training resolutions, and a mixed-public CUDA/recovery check. Old IPIC CUDA or data-capacity checks do not certify this different DINO graph. No such broader fit has been launched. This preparation does not require inventing physical quartets, but it also cannot validate their proposed mechanism.

At 13:24 KST, the new local FIRC reachability audit passed all **24,099 checks**: 8,033 native training rows at long sides 672, 784 and 896, with stride-four candidates and the actual ±16-pixel offset range. Maximum target counts are 1,912 for FSC and six for native D2S. D2S all-instance capacity is a conservative check for its category subsets, not permission to use all-products labels. The input manifest matches its original SHA256; no model forwards or optimizer calls were made. This audit checks label geometry and maximum assignment cardinality, not transformed appearance, CUDA training or generalization. The earlier image-byte/decode audits remain separate evidence; images were not decoded again here. See `artifacts/firc_public_readiness_v1_20261106/readiness.json`.

## Evidence

- Completed assignment analysis: `artifacts/firc_assignment_probe_v2_20261106/analysis.json`
- Preserved v1 failure: `study_protocols/firc_assignment_probe_v1_engineering_failure_20261106.json`
- Frozen LR plan and launch: `study_protocols/firc_lr_continuation_plan_20261106.json`, `study_protocols/firc_lr_continuation_launch_20261106.json`
- Current run: `artifacts/firc_lr_continuation_v1_20261106/`
- Reserved pool and deployment contract: `artifacts/firc_final_evaluation_readiness_v2_20261106/pool_and_contract.json`

The initial reserved-pool inventory omitted video-SHA ancestry and found only four groups. Its output is preserved; version 2 corrects that mapping and includes all five groups before any scoring.

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 14 files under the folder firc-assignment/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
