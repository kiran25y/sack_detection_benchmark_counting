# Source fidelity audit and one controlled detail repair

Owner authorized the source-training/failure-review plan and the subsequent justified repair. All eight V8/V9/V10 arms completed before this cycle. Their releases, checkpoints, labels, failed attempts and unrelated DGX jobs remain preserved.

## Fitting-only audit

The one-CPU FP32 audit used96 fixed source-fitting examples:32 at counts<25,32 at25-149 and32 at>=150. Seed2027 sampling excludes all1,066 class-held selection rows from the2,587-row fitting partition. It used the existing full-refit native-partial checkpoint for diagnostic interventions and gradients only, with zero optimizer steps. That checkpoint has seen all source training classes; these are not unseen-class accuracy measurements.

| Fitting stratum | Queries | Any exemplar side <2 cells at64 | Mean fine/coarse reference cosine, first level | Mean raw objective gradient norm |
| --- | ---: | ---: | ---: | ---: |
| <25 | 32 | 0 | 0.8142 | 774.70 |
| 25-149 | 32 | 6 | 0.7400 | 2485.83 |
| >=150 | 32 | 14 | 0.6284 | 13942.07 |

The original/native reference path and coarse query-grid reference path differ increasingly in the inspected dense group. At identical checkpoint weights, changing references increases the middle-group fitting-probe MAE from4.9995 to7.6552; dense from61.1471 to63.5061. This is a corrupted-input/intervention diagnostic, not a trained control or causal explanation of the previous full-fit gap. Raw gradient norms do not account for minibatch cancellation, clipping or epoch exposure; they do not support blindly treating dense gradients as weak. Normalized point-bin collisions remain geometric proxies and are not sufficient explanations for all errors.

## Failure candidates and annotation conventions

A separate CPU audit computed point-compatible review candidates on the37 previously identified shared-error source-validation queries. It does not filter predictions or alter scoring. Tags distinguish uncovered native point locations, boxes containing multiple native points, multiple singleton-point boxes and boxes without native points. Each is a review candidate: box overlap, displacement, incomplete annotation and counting-unit differences can produce the same signature. Semantic wrong-object counts cannot be established from a single foreground category ID alone.

Official `instances_val.json` image names/IDs match all1,286 source queries, but native FSCD box cardinalities equal the native FSC point counts on834 and differ on452. Original files and both definitions remain intact. FSC point-count evaluation and FSCD native-box AP must be named separately; no invented alignment, label replacement or missing-object verdict is introduced. This issue does not change the existing 8.8367/9.4168 point-count baseline scores.

Three fixed source witnesses were rendered with green native points and blue saved boxes in `artifacts/source_detail_v11/failure_witness_gallery.png` and visually inspected. The book image840 shows both broad multi-spine boxes and dense overlapping proposals. The marked-shirt image3483 shows many native point locations without a corresponding detection despite some broad boxes covering multiple points. Folded-clothing image7656 includes very thin repeated boundaries, CountGD++'s900 global-query saturation and many broad proposals. These are local qualitative signatures, not independently recounted labels or population-level failure rates. Native counts are637/261/1,231 respectively. The initial witness JSON write encountered a container-owned directory before any bytes; a fresh writable sibling recovered it without permission changes, preserving the failure receipt.

## Selected bounded repair and control

The fresh immutable `detail-counter-v11` keeps the same frozen SAM2 encoder,1024 input, native3x3 positive exemplar fields,470,359-parameter native-partial head, objective, AdamW optimizer, source class-held selection and recovery protocol. It retains query features at128/64/32 instead of64/32/16. The original cached `p0/p1/p2/mask/geometry` fields are copied byte-for-byte into the new cache; no degraded query-derived exemplar replacement is used.

| Serial arm | Shared query grids | Matching grid | Output/target grid |
| --- | --- | --- | --- |
| detail_control_native_partial | 128/64/32 | 64 | 128 |
| detail_partial_joint | 128/64/32 | 128 | 128 |

The candidate/control distinction is matching resolution. Parameter capacity, references, data and losses match; compute differs and selected refit durations can differ. Compare both with original64-grid native-partial and native-unit, which are pipeline comparisons with additional resolution/initialization/receptive-field differences. Native point labels are rasterized directly at128, never interpolated density labels. Every target sum must equal its native count. No augmentation, cropping, new backbone, negative prompts, target supervision, unit scaling or extra seeds are added.

Both arms select epochs on the unchanged source-training class holdout:cap60, patience12 after minimum20 epochs, seed2027. Each refits all3,653 eligible source-training images and then evaluates all1,286 reused native validation queries. Every completed optimization epoch atomically saves model, optimizer, RNG and permutation-generator state with strict code/data/cache identity. Failed attempts and uncommitted cache files are preserved; indexed feature files are atomically committed and validated on recovery. No original output is overwritten.

This is a resolution diagnosis/repair, not an established new mechanism. GeCo2 already uses high-resolution scale-specific queries; CountingDINO/LOCA/SAFECount already condition on exemplar geometry/similarity. A useful gain would justify closer mechanism controls, additional seeds and audited independent evaluation, rather than a novelty claim from finer grids alone.

## Executed preflight and current launch

Eight fixed source-fitting images (four sparse and four dense with small exemplars) were re-encoded.128 sampling improved point-location feature fidelity on all eight relative to64. For the four inspected dense examples, mean cosine distance to the native finest feature field fell from about0.4103 to0.1530. Labels chose measurement locations only; they were excluded from encoder inputs. This is a limited representation measurement, not trained counting accuracy or proof about all images.

Actual CPU/CUDA FP32 feature-relative-norm guard <=0.003, BF16 query-feature cosine >=0.995, same-input CPU/CUDA FP32 head tolerance rtol0.003/atol0.01, finite BF16 gradients and model/optimizer/RNG recovery passed for sparse/dense examples in both arms. The default64 matcher exactly reproduced the immutable original forward on the checked fields. Native reference bytes remain fixed. These are engineering checks, not trained accuracy or full-batch/per-image parity. Local focused tests passed16 checks using standard CPU kernels; the existing Windows oneDNN primitive issue and initial test-discovery failure are recorded separately, with no production-environment change.

Release has63 hashed Python files:54 prior V10 files preserved byte-for-byte, two existing math/controller files extended for the candidate, and seven added modules/tests including the fresh assessment module. An initial receipt conservatively reported53 preserved files because the new assessment module was subtracted as an old file; the original receipt is retained and corrected metadata explicitly records54. No executable bytes changed after the numerical guard.

`full-detail-source-v11` launched after V10 and preflight containers exited successfully. It is first building the full source cache, then trains the control and candidate serially. Launch checked >100GiB disk reserve and81.35GiB available host memory before the24GiB container. No additional heavy research GPU model overlaps it. Fresh CPU-only `full-source-decision-v4` watches10 arms. Its initial eight completed models, baselines and image/category paired comparisons exactly match preserved V3.

V4 validates checkpoint/prediction/native-query hashes and coverage, MAE/RMSE/bias/GAME, sparse/dense and class-macro errors, and paired image/category-component intervals immediately after each arm. These groupings do not prove scene independence, and reused validation is not independent confirmation. Reserved/external targets stay untouched. A useful controlled benefit with acceptable sparse/spatial costs is required before repetition or external evaluation. No trained V11 accuracy result exists yet.


## October 15: detail control complete; fine matching candidate refitting

Both source caches are complete (3,653 training and 1,286 validation queries). The V11 detail control completed 32 selected all-source refit epochs and all 1,286 reused native validation queries: MAE **14.4101472620**, RMSE **64.8299024751**, bias -8.0723979083, GAME(1)/(2) 16.0383075720/18.8938967851 and class-macro MAE 13.8572471496. Training MAE 5.5081471742 and internal class-held MAE 9.3158324668 are separate from native accuracy.

Control-minus-original-native MAE is +0.0079424842, with image-group CI [-0.7895009935,0.8345524677] and category-component CI [-1.4119352385,1.7856754299]: no reliable gain. It clearly loses CountGD++ global (MAE 8.8367) and GeCo2 global (9.4168). Dense 115-query MAE is 107.6122, contributing 66.78% of total absolute error. The control alone does not determine whether finer matching helps.

The candidate `detail_partial_joint` is actively refitting all 3,653 rows for 56 selected epochs; recoverable refit **6/56** was observed with no failure or OOM. Its native validation result remains pending. CPU watcher V4 has assessed 9/10 arms. Let this fit and evaluation complete before any new intervention, repetitions or external scoring. Selected durations differ (control32, candidate56, original40); pipeline comparisons are not fixed-checkpoint causal proofs. Reused validation and image/category groups do not establish independent scene confirmation. See complete control receipt.


## October 16: fine matching gain reviewed; automatic follow-ups resumed

V11 completed both arms and exited successfully; V4 assessed all10 arms. Fine matching selected/refit56 epochs and evaluated all1,286 native source-validation queries: MAE **12.4936851324**, RMSE **52.7660160755**, bias -1.1630913737, GAME(1)/(2) **14.1883154692/17.0851767380**, class-macro **11.9027882544**. Training MAE5.1873072607 and internal class-held MAE8.0632486965 are separate from native accuracy. Checkpoint `b68575c2f8a5db826eed1964dffbccc7d98449823a19822fba5578185c3c6af5`; predictions `874a7a75a85137e8520793918ad9720a002c25bc38bbdab32ed75d3e2786ea18`.

Candidate-minus-detailcontrol MAE -1.9164621296 has image-group CI [-3.4068241408,-0.5391542309] and category-component CI [-4.2902283755,-0.2498290586]. This supports a pipeline gain over control14.4101, with selected durations56/32 differing. Candidate-minus-original-native -1.9085 has image CI[-3.3800,-0.4126] but category CI[-4.6560,0.0786]. Candidate-minus-nativeunit -1.4046 has both intervals spanning0. The new point-estimate best direct model is not reliably better than every prior model under both grouping schemes.

It still clearly loses CountGD++ global8.8367 and GeCo2 global9.4168: differences +3.6570/+3.0769 have positive image and category intervals. Dense115 MAE86.1754 improves over control107.6122, while sparse603 MAE2.5605 versus2.4597 worsens descriptively. Dense errors contribute61.68% of total absolute error; aggregate bias near zero is not uniform accuracy. Reused validation is not independent confirmation; image/category groups do not establish scene independence or occlusion labels.

The conditional host queue correctly stopped at `deferred_for_gain_review`. After reviewing the useful gain and remaining baseline gap, the original deferral was preserved and the existing verified CPU audit was launched once as `source-representation-audit-v12`, with no GPU allocation or optimizer steps. It will diagnose remaining representation/decoder errors on48 source-fitting examples per checkpoint; preserve fine matching rather than discard its gain. Select only one justified matched source-only adaptation after the evidence is available. Existing adapters already train, and point-location ranking is not semantic negative labels or instanceAP.

The owner explicitly resumed automatic follow-ups. Existing heartbeat is now ACTIVE at its unchanged two-hour cadence, superseding prior paused-status notes. Surface meaningful comparisons, diagnoses/changes, failures or required decisions; stay quiet on unchanged checks. Read complete candidate result and audit launch. No adaptation pair or external target scoring has launched yet.
