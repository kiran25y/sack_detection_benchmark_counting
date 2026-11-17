# Counting direction decision — October 29, 2026

**Latest diagnosis, October 30 at 11:47 KST: saved-output audit completed; no new training.** Exact CPU replay reproduced all 810 saved selections. The learned gate and final box NMS remove native extreme-count hypotheses; moving surviving hypotheses back to native coordinates barely repairs spatial error. A new, shared-decoder point-set-supervision test is specified, not implemented or launched. Only three unused dense non-fitting FSC rows remain under the current exclusion policy, and all seven training extremes are consumed. Independent confirmation remains a separate requirement. The current correction heads use FSC training data only; historical sack data were used in V14, not this pilot. See the October 30 diagnosis and data sections at the end. The completed pilot's failed gate is unchanged.

**Latest verified result, October 30 at 11:22 KST: pilot COMPLETED; scientific gate FAILED.** All four containers and controller exited zero, no OOM; 1,580 counter forwards, six ten-epoch small-head fits totaling 1,440 updates, eight disposable engineering updates and zero counter updates were verified. On the 128-image correction-head decision cohort, mean-seed MAE improved from native GeCo2 9.164063 to correction 8.776042, but the 0.388021 gain was below the frozen practical threshold. Dense MAE improved 23.250000 to 21.947917; consumed extreme stress MAE worsened 69.285714 to 82.238095. Overall RMSE and spatial protection also failed. Fourteen of 39 gate components failed. This is not a standard FSC validation result or evidence of novelty. Do not launch a full fit from this result. The active/progress notes below are retained history; final details appear at the end of this document.

**Latest execution, October 29 at 19:38 KST:** owner authorized proceeding; the single `exemplar_jackknife_v1_20261029` controller is launched on DGX. It runs CUDA preflight, four-response capture, six fixed-duration small-head fits, and frozen analysis serially. Scientific gate pending; no full counter training or novelty claim. Frozen protocol: `artifacts/exemplar_jackknife_v1_20261029/protocol.json`, SHA256 `6e2b0d55ba46f548173c43fb5234e6a5001d085538bff84a362ebf393532ad5f`. The preparation/proposal-only status below is historical. Do not launch a second pipeline.

**Verified observation, October 29 at 19:50 KST:** CUDA preflight completed all 16 counter forwards and eight disposable scorer updates; native/repeat parity, finite gradients, numerical CPU/CUDA parity, pre-update restoration and numerical post-step replay passed. Frozen counter parameters remained unchanged. Bank capture is running, with 59 complete images and the 238th of 1,564 bank forwards in progress. Scorer training has not started. This observation is not a completion claim. Recent capture timings suggest roughly another 45–60 minutes, conditional on shared-GPU load; six small-head fits and verification follow. Completion heartbeat `geco2-jackknife-pilot-completion` is active every ten minutes, quiet for routine progress, and must be deleted after one verified final result or terminal failure report.

**October 29 reviewer-feedback update:** the review and new saved-prediction hybrid audit are recorded below under “Reviewer-feedback reconciliation.” The current recommendation is one grouped, fixed-duration test of local instance correction from GeCo2 exemplar-subset responses. Retire repeated one-step screens as go/no-go predictors for full training, preserve the old failures, and do not use test scores to select the direction. The earlier two-pass/512-update proposal below is historical preparation; the revised four-response pilot is still a proposal requiring a fresh protocol freeze, not an active job. No new inference or optimization was launched in this review.

**Recommendation: investigate local instance correction under changes in exemplar evidence. Park the image-level clarification predictor and further V11 loss/adapter variations.** The proposed contribution is a counter that changes individual object hypotheses using the before/after response to the same-target exemplar, with explicit one-to-one instance accounting. This is the strongest hypothesis supported by the evidence reviewed here, not an established novel method or a positive experiment result.

Status: **proposal only; no new model inference or optimization launched**. The work performed for this review was reading preserved results, checking primary prior work, and a CPU-only geometric analysis of existing predictions. The October 28 combined clarification gate remains FAILED. This proposal asks a new, explicitly declared three-exemplar counting question; it does not turn the failed request-policy gate into a pass.

## Why this is preferable to another loss or global predictor

| Measured evidence | Consequence for the decision |
|---|---|
| Retained V11 validation MAE 12.4937; matched CountGD++ global 8.8367 and GeCo2 global 9.4168 | Start from a strong pretrained detector for the new question; another full custom-head run is not justified by the current evidence. |
| V13 matching supervision 15.8180, V14 mixture 15.7200, V15 instance-mass 14.7626 on the reused 1,286-query cohort | Added supervision/data did not solve the measured problem. V14 also changed source exposure; these experiments do not isolate one universal cause. |
| Count-only pilot dense MAE 72.4093 → 90.1553 and GAME3 21.8574 → 32.4597 | Removing spatial supervision is not supported. These pilot scores are a different cohort from the 1,286-query table. |
| Ordered exemplar parts, frozen-base residuals and centered residual decomposition all failed their practical gates | Neither richer part inputs nor isolating a small spatial residual has demonstrated a useful protected gain. Late-state bounded failures do not prove impossibility from fresh initialization. |
| Fixed-scale GeCo2 two → three exemplar responses helped 41, hurt 34, left 53 unchanged among 128 development images | The useful evidence is mixed at the object level; an image-level decision discards that possibility. The existence of useful local edits still needs testing. |
| Fixed-scale GeCo2 oracle MAE 21.3047 → 16.4063 at up to 25% requests | This is truth-based headroom, not a deployable gain, not a population score and not evidence that a request predictor will work. |

The existing evidence does not support claims that all dense errors come from feature collisions, loss-gradient conflict, optimizer momentum, or prompt ambiguity. In particular, CountGD's observed global 900-query ceiling is a configuration-specific limitation, not an explanation for V11's errors.

## New descriptive check performed for this review

`scripts/audit_prompt_instance_changes_20261029.py` reads the preserved prediction banks, checks their saved hashes and computes maximum-cardinality bipartite box matches at both IoU 0.3 and 0.5. It used all 128 images, zero new forwards and zero optimizer steps.

For fixed-scale GeCo2 at IoU 0.3, the 56 dense images have, on average, **10.59 unmatched old boxes and 18.93 unmatched new boxes**, versus an absolute net count change of **18.23**. Both sides have unmatched boxes in **45/56** images. About **38.2%** of unmatched-box changes cancel in the net count; at IoU 0.5 this is **48.0%**. All seven extreme images have changes in both directions.

This is evidence that scalar count differences conceal substantial changes in spatial predictions. It is **not** proof of true object births/deaths, correct correspondences, or a repair mechanism: box size/localization shifts can also break a match. Cancellation is similar in helpful and harmful fixed-scale cases (38.0% versus 37.6% at IoU 0.3). Consequently, a raw disagreement/churn score is not the proposed solution. Local visual evidence has to add something beyond it.

Full descriptive results: `artifacts/prompt_instance_review_20261029/geometry_audit.json`. Both cutoffs are reported; neither is selected as a candidate's favorable threshold.

## Closest prior work and the actual novelty boundary

| Primary work reviewed | What is already covered; implication |
|---|---|
| [DAVE, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Pelhan_DAVE_-_A_Detect-and-Verify_Paradigm_for_Low-Shot_Counting_CVPR_2024_paper.pdf) | Candidate detection followed by exemplar-based verification. Adding a verifier alone is insufficient. |
| [GeCo2](https://arxiv.org/abs/2511.08048) | Scale-specific exemplar processing and high-resolution query aggregation. Better scale handling or a SAM-based counting head is not our contribution. |
| [CountZES, version 4](https://arxiv.org/html/2512.16415v4) | Training-free exemplar discovery/selection using complementary cues. Selecting good exemplars or adding feature consensus is insufficient. |
| [GIPACount, 2026](https://www.sciencedirect.com/science/article/pii/S0893608026011743) | Exemplar/context interaction, prototype aggregation and category-contrastive supervision. Generic contrastive loss, concatenation or prototype refinement is not a clear new direction. |
| [UniCounting, October 2026 preprint](https://arxiv.org/abs/2610.08379) | Learned same-instance relations and proposal consolidation into counts. A relation MLP, graph grouping or one-component-one-count rule alone is already closely covered. Its setting is fixed-vocabulary image-query-free counting, different from ours; that difference alone does not establish novelty. |
| [SAM 3, ICLR 2026](https://arxiv.org/html/2511.16719v2) | Exemplar-conditioned instance segmentation and interactive additions/removals are already supported. Merely switching to SAM 3 or calling the method interactive is insufficient. Its reported counting study uses different, low-count benchmarks and an internal checkpoint; those scores are not a matched dense FSC result for released weights. |
| [ScrewCount, 2026](https://www.scitepress.org/publishedPapers/2026/142363/pdf/index.html) | Exemplar efficiency and robustness, including non-monotonic exemplar-number results. “More exemplars can hurt” is not a new observation by itself. |

The narrow candidate distinction is **using the local response to a controlled change in exemplar evidence to decide which object hypotheses should be retained, removed or added, while preventing duplicate/many-to-one counting across those responses**. It must outperform both ordinary fusion and a verifier that lacks the proposed correspondence constraints. Bipartite matching and set-prediction losses themselves are established tools, not a novelty claim. This focused search is not exhaustive novelty clearance.

## Candidate mechanism, stated concretely

All three original native exemplar boxes are available to every method from the start. A deterministic, label-independent rule selects an initial pair. The pretrained counter produces two sets, P2 and P3, at exactly the same declared image scale. Its encoder and existing weights remain frozen.

1. Construct candidate correspondences between P2 and P3 using native-coordinate geometry and frozen region features. Enforce one-to-one correspondence: neighboring instances from the same response cannot be collapsed by transitive graph chaining.
2. A small shared scoring head uses local region/exemplar features, both native confidences, and the change in exemplar support to score retention, deletion and insertion. It must be able to reject a stable false positive and accept an object visible in only one response; simple agreement is not a truth label.
3. Decode an explicit corrected instance set. Each accepted correspondence contributes one location; accepted unmatched proposals contribute one each. The count is the cardinality of that set, and spatial metrics use those exact locations.

Conceptually, `count = retained matched instances + accepted old-only instances + accepted new-only instances`. There is no global multiplier, predicted count offset, image-level deferral, or ground-truth filtering at inference. Adding a positive exemplar is **not** assumed to monotonically increase the correct predicted count. Misses absent from both candidate sets cannot be recovered by this mechanism; candidate recall is an explicit early stopping check.

Only native training points supervise instance assignment and the shared set objective. Point-assignment correspondences are training surrogates, not object masks or measured instance AP. Ambiguous assignments are reported. No GT-derived instance scale, count, category or matching is available at inference. This design intentionally changes the prediction representation; it does not remove V11 spatial terms and silently call the resulting fit a matched V11 experiment.

## One decisive bounded test: local correction versus fusion and verification

This is a **proposed protocol requiring an input/code freeze before execution**, not a launched or already frozen experiment. The purpose is to falsify the local-edit hypothesis within a 48-hour decision window, not to establish final paper accuracy.

### Data and execution cap

- Propose 384 FSC training images: 192 fitting, 64 calibration and 128 decision images, grouped by native category plus known duplicate/provenance links. The decision set must have at least 32 sparse and 32 dense images and at least eight category groups; retain both medium strata. Select from metadata before new predictions. Check feasibility before claiming these exact groups exist.
- The previously examined 128 clarification images and all prior diagnostic rows are development evidence. Exclude them from the new decision set. Previously pretrained counters can have seen FSC training data; these are held-out groups **for the correction module**, not independent generalization of the underlying counter. The repeatedly inspected 1,286 validation queries stay closed during this decision.
- All seven FSC-training extreme examples have already been consumed, and the previous 1,066 class-held set had no 900+ examples. Retain all seven as a separate development stress check, exclude them from correction fitting/calibration, and never label them an untouched extreme test. A positive stress result cannot establish independent extreme generalization.
- At most 768 GeCo2 dataset forwards (two per new image), 14 extreme-stress forwards and 16 engineering/parity forwards. Reuse compatible valid caches. Extra feature capture must preserve author predictions within the frozen guard; no resizing, query-cap or checkpoint changes.
- Two declared seeds, 256 small-head update calls per seed; maximum 512 small-head updates. Frozen backbone/counter, no full counter fit, no seed replacement, coefficient sweep, checkpoint search or trajectory extension. A common scorer can be reused between decoder controls to isolate decoding rather than introducing extra trained capacity.
- Before launch, freeze the association cost/gating, feature definition, at-most-20,000-parameter head, shared point-set supervision, AdamW recipe, clipping, data order, both seeds and analyzer. This proposal does not invent a completed implementation or parity pass.
- Existing memory/disk and ownership guards apply unchanged. If input semantics, category coverage, resource admission or parity fail, stop and report the blocker rather than filling the cohort with favorable images or extending the time budget.

### Required controls on the identical inputs

| Arm | Purpose |
|---|---|
| A: native three-exemplar GeCo2 | Establish whether the proposed extra computation improves the real starting point. Report its lower actual inference cost. |
| B: ordinary geometric box fusion of P2/P3 | Test whether a conventional two-pass ensemble explains the benefit. Freeze fusion settings using calibration only. |
| C: unconstrained local verifier | Use the same pretrained features, candidate pool, trainable scorer/weights and supervision as D, but ordinary independent acceptance plus the same conventional deduplication as B. This is a DAVE-inspired control, not a claim to reproduce DAVE. |
| D: proposed local edit decoder | Add the one-to-one response correspondence and explicit retention/insertion/deletion accounting. Every other learned component is shared with C. |

A fixed correspondence-destruction ablation replays D with within-image shuffled cross-response links, preserving the number of proposals and global count/confidence summaries. This is an inference ablation, not a further trained arm or a replacement winner. If D gains equally with destroyed correspondence, the proposed local mechanism is unsupported.

All two-pass arms pay for both model passes, feature extraction and decoding. Native three-shot A remains a cheaper reference, not a fictitiously compute-matched method. Scalar averaging of the two counts is also recorded as a cheap count-only baseline; it cannot satisfy a localization claim without a declared spatial output.

A contemporary system reference is released SAM 3 with the same three positive exemplars and no extra text/negative prompts. Allow at most 64 calibration + 128 decision + 7 stress + 8 compatibility forwards, only if checkpoint access and native semantics can be verified within this same window. Treat it as a separate pretrained-system comparison, not matched capacity or equal training data. If unavailable, explicitly leave modern-baseline readiness incomplete; do not substitute its paper numbers or claim current state of the art.

### Prespecified decision

For **both** scorer seeds, D must satisfy all of the following on the unchanged decision set:

1. Overall MAE improves by at least `max(0.5 objects, 5%)`, and dense-150–899 MAE improves by at least 5%, against **each** of A, B and C. Beating only the custom V11 score is insufficient. Do not switch to a favorable winning arm.
2. Overall RMSE, sparse and each medium stratum's MAE/RMSE, and all/sparse/medium/dense GAME1/2/3 worsen by no more than `max(0.1 objects, 1% of that comparator)`. Report signed costs even inside tolerance. No count gain obtained by discarding hard images.
3. The paired category-group 95% interval for D-minus-control overall MAE has upper bound below zero against B and C. An inconclusive interval does not pass. Image/category grouping is not proof of independent scenes.
4. D beats the fixed correspondence-destruction ablation by at least 2% dense MAE without violating the same protections. Native-point matched recall/precision and edits of initially matched/missed points are also reported to show where the gain arises; do not claim AP from point labels.
5. All seven consumed extreme stress images are reported, with at least 5% aggregate extreme MAE gain over A/B/C and the same extreme RMSE/GAME protections. This is necessary stress evidence, explicitly not an independent extreme claim.

Any required component failing or being unevaluable means **do not proceed to a full counter run**. A failure would distinguish insufficient candidate recall, gains explained by simple fusion, gains explained by generic verification, and localization/sparse regressions. It would not prove that every possible instance-edit method is impossible. A pass justifies a separately planned confirmation study and full prior-work comparison; it is not automatic permission for another full fit or a paper claim.

## Time decision and deliverables

Allocate approximately six hours to data/protocol/parity readiness, up to twelve to bounded inference, and the remaining window to the tiny-head fits, all controls and independent scoring. **48 hours is a stop/decision cap after work is authorized and resources are admitted, not a measured runtime or a promise of success.** If implementation or semantic readiness cannot fit, report that limitation rather than spend another week collecting diagnostics.

CVPR 2027's [official dates](https://cvpr.thecvf.com/Conferences/2027/Dates), checked October 29, are November 10 registration and November 16 manuscript submission, 2026, AoE. The remaining time argues for one falsifiable direction. A passing pilot would still need external confirmation, modern baselines, reproducibility and writing; acceptance and novelty cannot be guaranteed.

Retain V11 and all negative outcomes. Do not restart V14/V15 or any old diagnostic. No reserved OmniCount, unresolved sack labels, public test selection, authentication changes or automation changes were performed in this review.

## Local evidence

- `docs/COUNTING_RESEARCH_REPORT_20261026.md`
- `docs/COUNT_ONLY_PILOT_RESULT_20261028.md`
- `docs/EXEMPLAR_PART_DECISION_20261028.md`
- `docs/FROZEN_RESIDUAL_DECISION_20261028.md`
- `docs/RESIDUAL_DECOMPOSITION_DECISION_20261028.md`
- `docs/CROP_POLICY_REVIEW_ARCHIVED_20261012.md`
- `docs/CLARIFICATION_FEASIBILITY_RESULT_20261028.md`
- `artifacts/exemplar_clarification_v1_20261028/analysis.json`
- `artifacts/exemplar_clarification_v1_20261028/independent_verification.json`
- `artifacts/prompt_instance_review_20261029/geometry_audit.json`

## Reviewer-feedback reconciliation — October 29, 2026

The main criticism is accepted: the current work has neither a competitive custom counter nor a demonstrated distinctive correction mechanism. More engineering checks cannot supply either. Repeated late-checkpoint one-step screens were useful for diagnosing local behavior, but were given too much decision weight and consumed time without predicting convergence. End that sequence. A future learned candidate should receive one fixed, short trajectory with matched controls and several seeds, followed by a single prespecified decision.

This update preserves all old results and gates. In particular, the combined clarification gate remains failed; GeCo2's two component passes can motivate a new GeCo2-specific question without relabeling the original experiment.

### Corrections to the supplied assessment

| Assessment or recommendation | Evidence-based correction / action |
|---|---|
| CountGD 5.74 test / 7.10 validation is the directly comparable headline | These are published text-plus-visual-exemplar results with inference enhancements; our 8.8367 is a measured CountGD++ visual-only global configuration. Show both with explicit input/compute/protocol columns. The custom V11 gap remains real. [P1] |
| The shared SAM2 framing proves the same architecture | GeCo2 already supplies a SAM2 counting head and scale-specific dense-query mechanism. Our generic framing has no distinctive contribution, but shared backbone alone does not prove identical models. A new mechanism must beat its own pretrained base and the closest controls. [P2] |
| Split the used validation set into A and frozen B | All 1,286 outcomes have already influenced development. A new partition cannot undo exposure. Use category/provenance-held groups from source training for correction-module selection, state that the base counter may have seen them, and audit a genuinely unconsumed cohort for final confirmation. |
| Run test now and use it to decide whether V11 survives | This is test-based selection. Baseline/crop work has already consumed FSC test evidence, although the custom V11 has not been scored there. A later frozen test result can be a transparent benchmark, not a newly pristine confirmation. Do not score V11 test to choose the current direction. |
| Three 10-epoch fits cost about one sixth of a full fit | On the same full data, 30/56 is about 54% of one 56-epoch fit per arm; subset fraction and actual feature/inference costs must also be counted. Fix optimizer steps, exposure, seeds and data order across learned controls. |
| Absolute count L1 is dominated by dense images, so log-count will repair it | Dense images dominate aggregate error, but d|S-N|/dS is only the sign, not the error magnitude. Parameter gradients depend on the model Jacobian and other losses. Log/relative count terms generally downweight large-count errors near the target; a dense rescue is not implied. Such terms and stratified sampling are established controls, not novelty. Do not open another loss sweep. |
| Many gates guarantee failure | The conditions are correlated safeguards, not independent 1% hypothesis tests; both GeCo2 oracle configurations did pass. Nonetheless the small-step tolerances and large condition lists were a poor proxy for training utility. Separate integrity checks from a few prespecified scientific endpoints, and retain sparse/spatial protection appropriate to the intended claim. |
| All three counters must pass future clarification tests | A model-specific GeCo2 study is valid if explicitly declared now. The CountGD global ceiling should be a limitation/control, not a veto over a new GeCo2-only hypothesis. This does not alter the old failed result. |
| Consensus predictions can automatically be kept as correct | Stable false positives and stable misses are possible. Consensus is a feature/control, not a truth label. The proposed correction must be able to reject stable errors and must disclose misses absent from every response. |

V11 stays archived as an experimental result and comparator. It is no longer the principal architecture to develop. The sack study remains separate; unresolved counts, foreground scope and recording separation prevent a benchmark claim. Choosing 500 images or five baselines alone would not solve those semantics.

### New completed analysis: hybrid oracle on existing validation outputs

Executed `scripts/audit_saved_hybrid_oracle_20261029.py` locally with standard Python only. It aligned all 1,286 unique query IDs, native labels and image groups, checked finite predictions, verified the V11 raw stream hash, cross-checked baseline stream identities with the independently retained audit, and required exact equality of every saved count across the two records. Original remote baseline files were not rehashed during this turn. A second direct replay from the independent audit and raw V11 stream reproduced both oracle MAEs. No new model forwards, optimizer calls, test scoring or GPU activity occurred.

| Prediction or selection | Overall MAE | 150–899 MAE | 900+ MAE |
|---|---:|---:|---:|
| V11 | 12.493685 | 59.623497 | 495.833897 |
| CountGD++ global | 8.836703 | 38.444444 | 368.571429 |
| GeCo2 global | 9.416796 | 50.916667 | 272.714286 |
| Truth oracle: better of the two detectors | 4.744168 | 23.046296 | 171.000000 |
| Truth oracle: better of detectors and V11 | 3.956696 | 18.086053 | 171.000000 |
| Fixed replay: CountGD, use GeCo2 only when CountGD predicts 900 | 8.360809 | 40.407407 | 250.857143 |
| Fixed replay: CountGD, use V11 only when CountGD predicts 900 | 9.228741 | 41.426801 | 394.580872 |

Oracle selection uses ground truth and is not a model result. The two-detector oracle has 6,101 total absolute error; the three-model oracle 5,088.311276. V11 is strictly better than both detectors on 218 images (65 sparse, 132 medium, 21 dense-150–899), but **zero of seven validation extremes**. It contributes 0.787472 overall oracle MAE improvement, all outside the extreme stratum. Therefore “V11 is the saturation-rescue expert” is unsupported, while “V11 never contains useful complementary information” would also be false.

The fixed saturation rule changes four predictions. GeCo2 reduces absolute error by 908 and 182 on two extreme images, increases it by 266 on another extreme, and increases it by 212 on a count-885 image. The overall gain of 0.475894 hides a dense-150–899 cost of 1.962963 (about 5.11%). Sparse/medium predictions are unchanged. This replay has no measured spatial protection and no independent evaluation; no threshold was swept. It motivates a strong simple routing control, not a new paper contribution.

The seven validation extremes here are **different images** from the seven training extremes in the clarification bank. GeCo2's 23% overall / 45% extreme clarification oracle improvement belongs to the stratified 128-image training bank and cannot be combined with these validation scores.

Complete reproducible evidence: `artifacts/reviewer_reconciliation_20261029/hybrid_oracle.json`.

### Current recommendation: one local-correction pilot, then a decision

Prioritize the proposed exemplar-jackknife experiment as a way to test local correction. Defer human-request policy learning until a correction mechanism itself is useful. Keep the hybrid oracle as evidence and simple routing as a control; do not split limited time across three independent method projects.

**Question:** do corresponding object hypotheses under controlled exemplar removal contain information that improves counting beyond ordinary ensembling and exemplar verification? The hypothesis is unproven. The 38% cancellation finding does not show useful signal by itself: cancellation was similar in helpful and harmful cases.

Use frozen pretrained GeCo2 and all three native visual exemplars. Run the full set and all three two-of-three subsets at the **same full-set-defined input scale and geometry**. No labels, truth-derived object sizes or extra text enter inference. Each pass uses unchanged author weights. Reuse old responses only if preprocessing, prompt set, weights and geometry match exactly. Four passes and feature/decoder overhead are charged explicitly.

The correction module scores candidate instances using local image/exemplar evidence and the response to exemplar removal. Its decoding enforces explicit one-to-one cross-response accounting to avoid duplicate counting. It can remove a stable false positive or retain a disputed instance; an intersection-only rule would bias toward undercounting. Report candidate recall because no scorer can recover an instance missing from the complete candidate pool.

**Proposed controls, all frozen before decision scoring:**

1. Native full-three GeCo2 (one-pass reference, cheaper).
2. Ordinary geometric fusion and a fixed consensus rule over the same four responses (simple ensemble controls).
3. An ordinary local exemplar verifier using the same candidate pool, pretrained features and matched scorer capacity; it does not use cross-response identity/change information. This is a verifier control, not a reproduced DAVE result.
4. The proposed response-aware correction with explicit instance accounting. A fixed shuffled-correspondence replay tests whether the response relationships matter. Raw native counts/confidences and candidate-set size must be preserved by that ablation.

The generic verifier and correction scorer receive matched initialization, optimizer, data order, head capacity, loss coefficients and update budget. No backbone fine-tuning, count multiplier or inference truth. Training point-to-proposal assignments require a label-semantics check: ambiguous close points are not silently converted to negative instances, and point-free background is not an instance mask. If reliable assignment is unavailable, stop before fitting rather than claim a supervised verifier.

**Proposed bounded design, not yet launched or feasibility-certified:** retain the 384-image grouped source-training split proposal (192 fitting, 64 calibration, 128 decision), exclude already inspected development rows from the decision set, and retain all seven consumed training extremes as a separately labelled stress set. Do not claim counter-unseen categories from correction-module-held groups. Group availability and duplicated-image provenance must be verified before freezing the cohort. Neither the reused native validation set nor FSC test is the pilot selector.

Replace the earlier 256-step/two-seed proposal with exactly 10 epochs for each of three paired scorer seeds on cached features: at batch eight and 192 fitting rows, 240 calls per fit, 720 per learned arm, **1,440 total** for the two learned arms. Use three final checkpoints, not best-of-epoch or best-of-seed selection. This tests a small correction head, not a full counter fine-tune. Four responses per 384 development images plus seven stress images and 16 engineering forwards cap new counter calls at **1,580**, reduced only by genuinely compatible cache reuse. These are proposed limits requiring a fresh immutable freeze; they do not authorize executing a new job from this review.

**Scientific decision:** primary endpoint is mean per-image absolute error averaged across the three scorer seeds, not count averaging into a three-model inference ensemble. Require at least max(0.5 objects, 5%) improvement versus native GeCo2, ordinary fusion and the matched verifier on the fixed decision cohort. Require a negative upper category-group 95% paired interval versus the matched verifier; disclose seed-wise variation. Dense-150–899 MAE must improve by at least 5% as the intended mechanism check. Prespecify protection for sparse MAE, overall RMSE, and all/dense GAME1 within max(0.1 objects, 1%); continue to report every stratum, GAME2/3, extreme stress behavior, edit precision/recall and material adverse effects. A smaller formal endpoint family does not hide a large harm on an un-gated metric. Do not claim extreme generalization from the consumed seven-image stress set. A dense/extreme repair claim requires at least a 5% stress improvement as supporting evidence, followed later by independent extreme confirmation.

One 48-hour decision window after code/data readiness and resource admission is a **cap, not a measured ETA**. If the candidate pool, implementation or guards fail, report the precise blocker. If gain is explained by simple fusion/verifier, the specific novelty hypothesis fails even if an ensemble improves MAE. If it succeeds, freeze the method and plan external confirmation plus current-system comparisons. No automatic full training, seed expansion or new mechanism follows a failure.

### Contribution boundary and nearest primary work

The intended distinction is instance-level use of controlled changes in exemplar evidence to decide which spatial hypotheses to retain, add or remove, with explicit cross-response counting. DAVE already verifies detections using exemplar similarity; generic verification is not new. ICACount already refines counting using human regional feedback; adding interaction is not new and its count-range supervision cannot be treated as equivalent to one box. UniCounting already learns proposal relations and consolidates instances; association alone is not new. SAM 3 already accepts exemplar concepts and produces instance masks. CoDi already targets dense small-object counting through exemplar-conditioned diffusion and has released checkpoint instructions. [P3–P7]

Therefore, “exemplar disagreement is uncertainty” or a four-pass consensus ensemble is insufficient. The response-aware mechanism must beat matched fusion and verification and lose its advantage when the proposed correspondence information is destroyed. A passing pilot would justify further work, not establish novelty, acceptance or current state of the art. CoDi and SAM 3 are contemporary system-comparison requirements; published numbers are not substitutes for compatible reproduced rows. Full CoDi PDF fetching was unavailable in this audit, so its assessment here is limited to its primary abstract and official repository, not a full-method reproduction.

### Primary sources rechecked in this review

- [P1 CountGD, Tables 1/3 and inference appendix](https://arxiv.org/html/2407.04619v2).
- [P2 GeCo2, architecture, comparison and limitations](https://arxiv.org/html/2511.08048v1).
- [P3 DAVE, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Pelhan_DAVE_-_A_Detect-and-Verify_Paradigm_for_Low-Shot_Counting_CVPR_2024_paper.pdf).
- [P4 ICACount, ICCV 2023](https://arxiv.org/abs/2309.05277).
- [P5 UniCounting, October 2026 preprint](https://arxiv.org/abs/2610.08379).
- [P6 SAM 3](https://arxiv.org/abs/2511.16719).
- [P7 CoDi primary abstract](https://arxiv.org/abs/2512.20153) and [official released-checkpoint instructions](https://github.com/gsustar/CoDi).

No report/PDF deliverables were recreated. This update modifies the existing research decision notes and saves a small reproducible analysis of already retained predictions. No remote processes, releases, labels, inference protocols, credentials or automations were changed.

## Authorized pilot execution — October 29, 2026

The owner subsequently requested proceeding and finishing the training and experiment. One pipeline was launched at 19:38 KST, controller PID 766095, remote job `research-direction-20261012/exemplar_jackknife_v1_20261029`, immutable release `releases/exemplar-jackknife-v1`. This authorization covers the bounded pilot and its matched controls. A positive pilot alone cannot certify novelty, independent confirmation or acceptance.

The frozen cohort has 192 fitting, 64 calibration and 128 decision images, plus seven consumed training extremes. It has 391 unique image hashes; fitting/calibration/decision/stress category components do not overlap. The decision set spans 27 category components and excludes 1,710 previously reviewed/selection-used query IDs, including the earlier 1,066-row class-held set, 128 clarification rows, fitting probes and recorded fidelity/failure reviews. The base counters may have seen these source-training images. Category/exact-image separation does not establish scene independence.

An initial hash-only category allocation could not meet fitting/calibration dense quotas (63/64 and 14/16). That preparation failure was preserved. Before any predictions, deterministic metadata-only component swaps achieved all original quotas. A subsequent exclusion audit found five decision rows in the old fidelity audit; the preliminary cohort was preserved and corrected before freezing. No outcomes were used for these selections, no scientific threshold was lowered, and native public splits/labels remain unchanged.

The native GeCo2 three-exemplar preprocessing defines one scale for all four full/leave-one-out responses. A passive backbone hook captures the same three frozen SAM2 feature levels; actual native prediction/repeat parity is required. Cached descriptors use fixed orthogonal projections of normalized ROI features, exemplar cosines, box geometry and native scores. Both learned scorers have exactly 4,161 parameters. The generic verifier uses local features; the correction replaces 16 local descriptors with 16 descriptors of cross-response support/confidence/geometry and uses explicit one-per-response associations. A within-response member shuffle preserves proposal coverage and per-response membership counts as the mechanism ablation.

Both learned arms use the same point-proximity supervision and per-image BCE objective, initialization, data order, AdamW (learning rate 0.001, decay 0.0001), clipping at 1, batch eight, ten epochs and seeds 2027/2028/2029. Each fit has 240 updates, totaling 1,440. This is a new instance-verification representation with shared supervision across controls, not a continuation or claimed isolated loss repair of V11. Point-proximity labels are not masks: matched proposals are positive, unmatched near-point proposals are ignored, and far proposals are point-location negatives. All assignments and ambiguous counts are retained. Threshold 0.5, association IoU 0.3, center-distance gate 0.5 anchor side and final NMS 0.5 are frozen; no threshold search or epoch/seed selection occurs. Calibration is report-only and does not enter fit statistics.

The engineering stage permits 16 GeCo2 calls and eight disposable head updates for CPU/CUDA, gradient and recovery checks. The capture bank permits 1,564 calls. No backbone optimization occurs. Three final seed results are scored as mean losses, not combined into an inference ensemble. Native, fusion, consensus, generic verifier, correction and shuffled-correspondence outcomes are all retained. The frozen analyzer reports every stratum and GAME1/2/3, primary/protection checks, per-seed costs and grouped uncertainty. The consumed extreme stress gate is necessary supporting evidence, not an untouched extreme claim.

Shared GPU use was explicitly authorized. Unchanged admission is 48 GiB available on two checks 30 seconds apart; stop floor is 32 GiB, disk reserve 100 GiB, container limit 24 GiB with no extra swap and two CPUs. The controller checks every 30 seconds, can stop only its own container, has a 48-hour cap, and never retries a failed stage. GPU containers exit before the CPU scorer-training stage. Existing V14/V15 and unrelated jobs remain untouched. Check live receipts for phase/progress; a launched controller is not evidence of completed optimization.

Completion collection and independent verification are prepared in `scripts/collect_jackknife_once_20261029.py` and `scripts/verify_jackknife_saved_20261029.py`. They use read-only DGX access and saved outputs, with no new inference, model loading or optimization. The collector waits for completed receipts and checks all four live Docker exits, frozen inputs, eight release files and 104 external source/weight hashes. The local verifier checks the original native labels, role/category separation, 391 finite feature banks, six ten-epoch histories, twelve complete raw prediction streams, all recomputed aggregate/GAME metrics, the grouped interval and every frozen gate component. It must preserve exclusive output archives and any verification failures.

Two limitations must remain explicit in interpretation. Equal capacity and optimizer budget do not make this a feature-only intervention: the correction head also uses group-based decoding while the verifier uses NMS directly. Likewise, the frozen shuffle changes both response descriptors and decoding groups. A positive result would support the complete correction pipeline over these controls, but would not isolate the causal value of correspondence features alone. The point labels are location surrogates, and the seven extremes are consumed stress cases. Neither a positive gate nor a new low MAE would by itself establish novelty or an independently confirmed CVPR contribution.

## Verified completed pilot — October 30, 2026

Read-only collection and independent local verification finished at 11:22 KST. The archived output SHA256 is `9bfa4865d1d259e22b3f77ba70e90094e8e37bfc52cde93b04343f8c6ec76521`. All eight release files, 104 external source/weight files, frozen input identities, 391 finite feature arrays and original native labels matched. All four Docker containers and the one-shot controller exited zero without OOM. Six final heads each completed exactly ten epochs/240 AdamW updates; total 1,440, plus eight disposable engineering updates and zero counter-backbone updates. All twelve raw prediction streams cover all 391 query IDs exactly. Independent MAE/RMSE/bias/GAME1/2/3 recomputation and the grouped interval match the frozen analyzer. No new inference, optimization or model deserialization was used in verification.

The run finished around 22:07 KST on October 29, estimated by correcting the DGX clock's approximately eight-hour UTC offset; the exact client verification time is authoritative. GPU response capture took about 142 minutes. The small-head training stage took approximately 74 seconds including its preparation and prediction work. These timings are shared-resource observations, not exclusive benchmarks.

The main cohort has 128 decision images, 32 per sparse/medium25/medium75/dense stratum, spanning 27 category components held out from the correction-head fit. Values below average the three seed losses for learned arms; they are not inference ensembles. The seven extreme images are a separate previously consumed training stress set.

| Method | Decision MAE | Dense 150–899 MAE | Sparse MAE | Decision RMSE | Decision GAME1 | Extreme stress MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native GeCo2 | 9.164063 | 23.250000 | 1.875000 | 18.414286 | 11.054688 | 69.285714 |
| Four-response fusion | 10.585938 | 29.031250 | 1.750000 | 26.826555 | 12.976563 | 165.428571 |
| Fixed consensus | 9.289063 | 22.562500 | 2.093750 | 18.793907 | 11.101563 | 87.285714 |
| Matched local verifier | 10.315104 | 27.989583 | 1.718750 | 25.874748 | 12.705729 | 74.904762 |
| Response-aware correction | 8.776042 | 21.947917 | 1.697917 | 18.943034 | 11.317708 | 82.238095 |
| Shuffled correspondence | 43.140625 | 100.843750 | 6.458333 | 60.918875 | 43.536458 | 657.238095 |

There is a measured effect, but it fails the declared decision. Against native GeCo2, overall MAE decreases 0.388021 (4.23%), below the required max(0.5,5%) gain. Dense MAE decreases 1.302083 (5.60%) and sparse MAE decreases 0.177083 (9.44%). However, overall RMSE increases 0.528748, overall GAME1 increases 0.263021, dense GAME1 increases 1.197917, and extreme MAE increases 12.952381 (18.69%). Extreme RMSE increases 20.997031 and extreme GAME1 increases 15.428571. Thus the failure is not merely an overlong checklist or a missing significance threshold: the intended extreme/spatial protection is violated.

All fourteen failed frozen components are retained in `independent_verification.json`. Against native: insufficient practical overall gain, overall RMSE, overall GAME1, dense GAME1, extreme gain, extreme RMSE and extreme GAME1. Against consensus: insufficient dense gain, overall GAME1, dense GAME1 and extreme GAME1. Against verifier: extreme gain and extreme RMSE. The final failed component is the paired category MAE interval versus verifier: correction-minus-verifier -1.539063, 95% interval [-3.526915, 0.117590], spanning zero. Every comparison against ordinary fusion passed. Improvement over shuffled correspondence passed, but shuffling also damages geometric grouping in decoding, so this cannot establish a feature-only causal mechanism.

The three correction-minus-native overall differences were -0.468750, -0.296875 and -0.398438 for seeds 2027/2028/2029. All improve numerically, none reaches the native practical-gain threshold. Against the seed-matched verifier, differences were -1.679688, -1.382813 and -1.554688; seeds are repeated fits, not independent cohorts. The candidate also improves on consensus overall by 0.513021, but its dense gain versus consensus is only 0.614583 (2.72%), below the 5% requirement.

Material secondary costs are not hidden. Versus native, medium75 MAE increases 8.250000 to 8.395833; overall GAME2/3 increase 14.945313/22.054688 to 15.421875/22.421875; sparse GAME1/2/3 increase 2.812500/3.875000/6.687500 to 2.927083/4.135417/7.135417. Dense GAME2/3 also worsen. Extreme GAME3 improves, while extreme GAME1/2 worsen. Lower aggregate bias (-3.039063 to -0.161458) does not undo these accuracy costs. Fitting assignment audit contains 74,199 positive, 5,416 negative and 2,382 ignored ambiguous proposal labels across the four dependent responses; these are not independent object labels.

Decision: preserve the complete negative result and native references; do not promote this correction to a full counter fit or a publication claim. The 8.776 pilot number must not be compared to the 8.837 CountGD or 9.417 GeCo2 numbers from the different 1,286-image native validation cohort. The base counter may have seen these source-training images, exact-image/category grouping is not scene independence, and stress extremes are consumed. C and D also differ in their decoder, so evidence does not identify the unique causal contribution of response descriptors. No further arm, threshold change, seed selection, full training or test-set scoring was launched after this result.


Completion follow-up `geco2-jackknife-pilot-completion` was deleted after verified-result reporting on October 30. All model outputs, checkpoints and historical receipts remain preserved.

## Failure diagnosis from exact saved-score replay — October 30

Evidence: `study_protocols/jackknife_failure_diagnosis_20261030.json`, `artifacts/jackknife_failure_audit_20261030/analysis.json` and `queries.jsonl`. All 135 decision/stress images and all three saved seeds were retained. Six existing 4,161-parameter scorers were replayed on CPU in their original runtime with restricted checkpoint loading; all 810 saved center selections matched exactly. No counter forward, optimizer update or original output modification occurred. A first NumPy replay exposed a near-tied representative differing by about 0.02 pixels; the original-runtime replay resolved it. Both that discrepancy and an earlier temporary-directory import failure remain recorded. The isolated runtime and local diagnostic dependency did not alter training environments.

The following are **post-hoc fixed-score counterfactuals**, not new trained methods or validation gains. Values average seed losses; images, not the three seed observations, are the sampling units.

| Saved-score decoding | Decision MAE | Dense 150–899 MAE | Extreme MAE | Dense GAME1 | Extreme GAME1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native GeCo2 | 9.164063 | 23.250000 | 69.285714 | 26.375000 | 99.857143 |
| Original correction | 8.776042 | 21.947917 | 82.238095 | 27.572917 | 115.285714 |
| Omit only final box NMS | 9.005208 | 23.072917 | 69.571429 | 28.114583 | 114.619048 |
| Keep original selected groups/counts; restore native centers where available | 8.776042 | 21.947917 | 82.238095 | 27.572917 | 115.190476 |
| Static verifier scores with the correction decoder | 9.325521 | 23.895833 | 76.095238 | 29.875000 | 124.571429 |

Final NMS helps moderate dense counts while harming extreme counts. Omitting it almost removes the aggregate extreme MAE gap but does not beat native GeCo2 or repair spatial accuracy. Restoring native centers leaves dense GAME1 unchanged: coordinate jitter is not the main measured spatial problem. Which hypotheses are retained or deleted matters more. The response scorer also worsens extremes against the static scorer under the same decoder (76.095238 to 82.238095), so the failure cannot all be assigned to NMS.

On `2347.jpg`, truth is 1,912; native count is 1,911; correction averages 1,724.33. Its gate rejects 242 native groups, final NMS removes another 1.67, and 57 new groups survive. Removing NMS still leaves absolute error 184.33. A nearly correct native total does not prove all removed hypotheses were true objects, but the edit accounting identifies the damaging computation. On `805.jpg`, correction helps substantially (absolute error 110 to 21.67); on `804.jpg`, it worsens 129 to 155. All seven cases remain in the receipt.

Across the seven extremes, the mean count changes from 1,329.86 to 1,292.90 through 90.43 native groups rejected by the score gate, 5.10 native groups removed by NMS and 58.57 new groups added. Groups contain at most one native detection: initial grouping does not merge two native anchors. The later gate/NMS can remove them.

Fitting counts span 8–645, whereas the stress set spans 953–1,912. Of valid training proposal labels, 93.2% are positive. The old target is already a maximum-cardinality, one-to-one point assignment **within each response**, with unmatched near-point proposals ignored. It is not joint supervision of the final combined instance set. Cross-response duplicate selection, far higher counts and a different final decoder provide a plausible supervision/generalization mismatch; this audit does not establish that as the sole cause. Point-assignment ambiguity is not verified semantic object merging.

## One next falsification test, proposed only

Specification: `study_protocols/point_set_correction_proposal_20261030.json`. Test whether **supervising the final unique-instance selection** enables exemplar-response evidence to improve safe local edits beyond static verification. Use the existing frozen pretrained GeCo2/SAM2 feature banks. All arms have exactly the same candidate groups, output centers, threshold, architecture and decoder. Native-containing groups retain their native center; other groups retain their fixed geometric anchor. Emit at most one point per accepted group and apply no final cross-group box NMS. This common decoder is a control, not the claimed candidate contribution.

| Arm | Features | Supervision |
| --- | --- | --- |
| A | Static local descriptors | Existing per-response assignment target inherited by the anchor, including its ignore rule |
| B | Static local descriptors | One-to-one point assignment across the final group anchors |
| C, sole candidate | Local plus real exemplar-response descriptors | Same final-set assignment as B |
| D | Local plus within-image shuffled response descriptors | Same final-set assignment as B/C; all geometry and grouping unchanged |

All heads remain 128→32→1 (4,161 parameters), use per-image BCE, AdamW 0.001/decay 0.0001, clipping 1, batch eight, ten epochs and three matched seeds. Twelve heads total 2,880 updates; no new counter capture or backbone updates for this cached development test. No threshold, seed, epoch or coefficient search. The A/B contrast changes target semantics, including whether unassigned near-point candidates are ignored or treated as point-selection negatives. B/C changes feature allocation at matched capacity, while C/D isolates response information without damaging the decoder. Set-assignment negatives mean redundant/unmatched point hypotheses, not semantic background masks. Ambiguous assignments and unmatched truth must be audited on fitting data before freezing executable code.

Proposed primary gate: C improves decision MAE by at least max(0.5 objects, 5%) against native and B, with the category-paired interval upper bound below zero against B. Dense and consumed-extreme MAE each improve at least 5%; protect sparse MAE, overall/extreme RMSE and all/dense/sparse/extreme GAME1/2/3 within max(0.1 objects, 1%). C must beat train-matched shuffle D on dense MAE by at least 2% without overall regression. Report medium strata, individual extremes and seed variation. These are proposed criteria for a new test, not a revision of the failed pilot's gate.

If B explains the improvement, there is no demonstrated exemplar-response contribution. If C fails protection, stop this correction direction rather than starting a full fit. A pass on consumed development data supports only a separately frozen confirmation study. The metadata audit finds 1,274 unused rows outside current fitting/stress category components, but only three are dense and none extreme. Do not construct a supposedly fresh balanced decision set or rename consumed rows as confirmation. A suitable additional cohort needs an exposure, label and split audit before any scientific confirmation claim.

### Prior work and scope of the possible contribution

One-to-one point learning is established by [P2PNet](https://arxiv.org/abs/2107.12746); learned verification by [DAVE](https://openaccess.thecvf.com/content/CVPR2024/papers/Pelhan_DAVE_-_A_Detect-and-Verify_Paradigm_for_Low-Shot_Counting_CVPR_2024_paper.pdf); learned suppression by [Learning NMS](https://openaccess.thecvf.com/content_cvpr_2017/html/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.html) and density-sensitive suppression by [Adaptive NMS](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.html). [UniCounting](https://arxiv.org/html/2610.08379v1) learns same-instance relations and consolidation in a different counting setting. These precedents rule out claiming a small verifier, graph, one-to-one target or changed NMS as new by itself.

The remaining hypothesis is that controlled changes of exemplar evidence provide useful instance-selection information beyond static appearance/geometry, under a common decoder and independent confirmation. Novelty remains unestablished. [VQCounter](https://www.ijcai.org/proceedings/2025/0252.pdf) also constrains point-matching/prompt-diversity claims, and [AgriCountDINO](https://arxiv.org/abs/2609.29460) constrains pretrained point-decoder and exemplar-adaptive point-NMS claims. Full [CoDi v2 HTML](https://arxiv.org/html/2512.20153v2) was accessible in this review, extending the earlier abstract-only reading: exemplar-conditioned latent diffusion for point-map counting is also existing work, not an unexplored replacement. None of these methods was reproduced in this diagnostic.

## Dataset and augmentation follow-up — October 30

The current correction pilot used 192 FSC training images to fit heads, 64 for report-only calibration, 128 for the decision and seven consumed extreme stress images. It used no sack images. These are correction-head roles; do not imply that the pretrained base never saw FSC training data.

V14 did use our sack data: 3,653 eligible FSC training queries + 4,000 prepared FSCD-LVIS training queries + 97 approved historical whole-image sack queries, total 7,750. Its FSC MAE worsened from V11's 12.493685 to 15.720033, while LVIS MAE improved from 39.295701 to 14.577309 on the same 1,181-query LVIS cohort. This demonstrates a domain tradeoff, not that additional data are inherently harmful. The fixed optimization budget reduced FSC presentations to 96,005, so domain mix and source exposure were confounded. The 97 sack rows have count-only labels and cannot supervise fictional instance points. The newer 79-frame foreground-truck review remains outside training/scoring until its scope, exact counts, prompts and recording separation are resolved.

Recommended data work: first audit the already prepared FSCD-LVIS training cohort for compatibility with the new point-set target and for split/exposure separation; it is not a newly untouched domain. Consider **MCAC native training data** for multi-category and occlusion coverage only after native labels, exemplars and visibility conventions are audited. Its author dataset has 1–300 objects per target class, so it does not supply independent 900+ evidence. Native MCAC test data and reserved OmniCount must not be moved into training. New foreground sack images require a separate, reviewed application-training cohort and recording-separated evaluation; raw frame count is not labeled data coverage.

Start augmentation with label-preserving horizontal flips, scale variation/padding and bounded color/exposure changes. Transform point coordinates and exemplar boxes with the same geometry; retain exact point cardinality and count semantics. Cropping requires explicit retained-point rules and intact usable exemplars; it must not silently retain an original whole-image count. Avoid random erasing, severe darkening and generic CutMix/MixUp that change visible membership while keeping old labels. Count-preserving dense composition is a later controlled training condition, not a novel idea or independent real-extreme evidence. [GeCo2](https://arxiv.org/html/2511.08048v1) already uses scale augmentation; [CounTR](https://arxiv.org/abs/2208.13721) already synthesizes high-count and multi-category training images. [MCAC's author repository](https://github.com/ActiveVisionLab/MCAC) specifies object metadata and native exemplar rules.

Do not add a new mechanism, dataset and augmentation simultaneously. First complete the shared-decoder target test on the existing cache. Then compare no augmentation against one frozen geometry/photometric recipe on identical fitting images, seeds and optimizer budget. Image augmentation requires new counter features/proposals; perturbing cached descriptors is not an equivalent implementation. Extra-data effects need a separate budget-matched comparison with the changed FSC exposure explicitly reported, plus an exposure-matched control if a causal data claim is intended. No added-data or augmentation fit has been launched by this diagnosis.
