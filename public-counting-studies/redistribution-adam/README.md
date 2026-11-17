> October 26, 18:00 KST verified: this diagnostic was STOPPED by its unchanged 32-GiB host-memory guard at 28.1783 GiB available. Docker exited 137, OOMKilled=false. Ten disposable calls, zero complete saved batch comparisons, no final report. The scientific gate is UNEVALUABLE, not failed or passed. Old evaluating status is stale after shutdown. No restart/full fit; outputs preserved. The completion monitor was deleted. See `study_protocols/optimizer_interference_stop_20261026.json`.

> October 26 update: the owner-authorized bounded optimizer-interference diagnostic is now launched; its gate is pending. See current protocol and status. The prior redistribution gate remains failed. No full training is authorized or launched by this diagnostic. Historical proposed-only wording below is preserved.

# Bounded redistribution Adam test: verified negative screen

**The frozen gate failed. Retain V11; do not start a full run of this candidate.** All 12 checkpoint/batch comparisons and 144 disposable Adam calls completed successfully. There were no persistent optimizer steps, checkpoint writes or GPU use. Execution took 10,564.31 seconds, approximately 2 hours 56 minutes. Docker exited 0 with OOMKilled=false and the one-shot controller exited normally.

The result was retrieved and independently verified on October 26, 2026, after 15:14 KST using client time. DGX wall-clock timestamps are about eight hours ahead of client UTC and must not be presented as client completion times. The failed first admission remains preserved.

## All prescribed dense comparisons

These are fitting-data MAEs after disposable updates, **not new validation MAEs**. The primary control applies the original V11 objective to the same checkpoint and optimizer state. The candidate adds the frozen anchored normalized allocation term. A negative delta is better. Repeated spillover and supplemental observations remain dependent.

| Starting state | Evaluation role | Unique queries | Primary MAE | Candidate MAE | Candidate minus primary | Relative improvement |
|---|---|---:|---:|---:|---:|---:|
| V11 | Dense update images | 16 | 23.773132 | 23.777368 | +0.004235 | -0.0178% |
| V11 | Dense disjoint spillover | 16 | 24.195205 | 24.200097 | +0.004892 | -0.0202% |
| V11 | Supplemental dense | 13 | 93.398419 | 93.392139 | -0.006280 | +0.0067% |
| V11 | Supplemental extreme | 7 | 115.703131 | 115.700152 | -0.002979 | +0.0026% |
| V15 | Dense update images | 16 | 12.168663 | 12.136601 | -0.032062 | +0.2635% |
| V15 | Dense disjoint spillover | 16 | 13.813744 | 13.827788 | +0.014044 | -0.1017% |
| V15 | Supplemental dense | 13 | 248.602153 | 248.714155 | +0.112002 | -0.0451% |
| V15 | Supplemental extreme | 7 | 320.190354 | 320.374468 | +0.184114 | -0.0575% |

There are small above-tolerance improvements in three cells, but **0/8 satisfies the required 1% relative improvement**. This is insufficient practical evidence, not a claim that every mathematical effect is exactly zero. The 0.2635% V15 update-image improvement does not carry to disjoint dense or extreme probes.

## Sparse and spatial costs, and fair controls

| Frozen family | Conditions passed | Outcome |
|---|---:|---|
| Dense/extreme gain | 0/8 | Failed |
| Sparse/medium count and GAME protection | 57/82 | Failed |
| Simpler and displacement-matched controls | 41/112 | Failed |
| Normalized spatial discrepancy | 26/48 | Failed |

These are dependent decision conditions, not independent statistical trials or estimated probabilities.

The V15-state sparse update MAE rises from 0.851166 to 0.853998 (+0.002832), exceeding its fixed 0.000115 numerical tolerance. V11 supplemental tiny-count MAE rises by 0.000168, exceeding its 0.000107 tolerance. Medium spillover MAE also worsens in both states. These are small absolute changes, but the prespecified non-regression requirement is not met.

On V15 supplemental extreme queries, GAME1/2/3 worsen by +0.185757/+0.193422/+0.149010. V15 dense spillover normalized-blur discrepancy rises by about 0.0000999. Some other spatial measures improve, so report the complete pattern rather than declaring universal spatial degradation.

The candidate beats the full-update-norm-matched primary control in only 1/8 dense cells, and beats the increment-norm-matched normalized-blur control in only 1/8. It therefore does not establish a useful effect specific to the exemplar/spacing allocation mechanism. Other raw arms have isolated gains, but there is no authorization to select a posthoc winner or treat an adapted DM objective as a reproduced author model.

## Verification

The independent analyzer checked all 12 unique state/batch outputs, arm coverage, native truth and GAME mass conservation, finite outputs, prescribed roles, exact zero-term replay, state/RNG preservation, Adam step 25,593 for each discarded update and OT marginal tolerance. Both starting states were finite at epoch 56 / Adam step 25,592 and matched their respective final model checkpoints.

All **172 independent file checks** matched: 101 immutable release files, 65 feature files, four original checkpoint files and the source manifest/class map. Frozen job hashes and report/batch result hashes also matched. The 65 unique fitting images exclude the source class-held partition; 222 observations per state include repetitions and are not 222 independent examples.

Files: `artifacts/redistribution_update_v2_20261026/result_outputs.zip`, `analysis.json`, and `decision_evidence.json`. The complete verified gate result is preserved, including every failed condition. No source-validation, external-domain or sack predictions were generated by this test.

## What can and cannot explain the failure

The candidate fails as a practically useful, protected late-optimizer update. This does not establish that it can never learn from fresh initialization, or that a different architecture is necessary. It also does not show that simply increasing its coefficient would work.

A concrete unresolved mechanism is **interference between the current count/spatial gradients and the optimizer's historical state**. In the two V11 dense batches, primary preclip norms are approximately 16,152 and 17,959, with clip factors 0.0000619 and 0.0000557. The candidate's additional parameter displacement is only 1.09% and 0.86% of the primary step norm. Corresponding V15 dense ratios are 6.03% and 4.22%.

These observations justify isolating optimizer effects, but do not prove that clipping or momentum caused the failed gate: Adam can compensate for gradient scaling, and some directions may simply be ineffective. The displacement-matched controls already show that increasing total movement alone is an inadequate explanation of the candidate's value. The earlier regional-projection experiment also achieved only tiny dense effects; generic gradient surgery is not an established repair.

## One next bounded diagnostic, proposed only

**Question:** Is the added spatial-allocation direction useful but distorted by historical first moments and global clipping, or does it remain ineffective when those factors are isolated?

Before another full fit, propose a small factorial optimizer-interference diagnostic, keeping the model, losses, native inputs and evaluation definitions fixed. This is a diagnostic of a mechanism, not a new loss or a proposed production optimizer change.

- Use both original V11/V15 states and the first predeclared batch of each stratum (existing batch indices 0, 2, 4), selected by original order, not by favorable results. Retain all 65 fitting images for evaluation, including all seven extreme queries.
- Compare only the already-defined primary and anchored-candidate objectives. For each, cross preserved versus zeroed **first** moments with original clip1 versus an unclipped diagnostic gradient. Preserve second moments, Adam step, learning rate, weight decay, model weights and all other state. These counterfactual optimizer states are disposable and must be labeled explicitly; they are not fresh training or recovery.
- Evaluate raw proposals and proposals matched to the original primary parameter-step norm. Include a zero-current-gradient control to measure historical-momentum/weight-decay motion, plus an exact primary replay. Six state/batch contexts give at most **60 disposable Adam calls**; all proposals reset and are discarded. No trajectory, coefficient sweep or new head.
- Record current primary/auxiliary gradient directions and their radial-count versus normalized-spatial directional effects before clipping, after clipping and after the proposed Adam step. Check finite differences and parameter-group contributions. Raw-gradient descent alone is not accepted as evidence of an actual finite-update improvement.
- A favorable result would require a consistent candidate-specific effect beyond the matched primary/null controls in both states: at least 1% dense and extreme gain, the same numerical sparse/medium and GAME protection, and spatial improvement. All rows and counterfactual conditions must be reported; a favorable optimizer setting cannot be selected after the fact. If only the primary benefits, or spatial effects remain tiny/adverse, the candidate is not rescued.

The intended causal comparison is the candidate-minus-primary effect under the factorial interventions, after controlling update norm. A positive interaction would support investigating count/localization update separation; it would still not justify a full run or novelty claim. A negative result would close this optimizer explanation at the tested states and motivate a separate frozen representation/decoder diagnostic rather than more auxiliary-loss trials. A null result cannot prove impossibility from scratch.

The next diagnostic has **not been implemented or launched**. Its runtime and numerical preflight must be established before execution; retain the prior two-CPU/16-GiB/no-GPU resource envelope, 48-GiB admission, 32-GiB host stop floor, 100-GiB disk reserve and four-hour bound if it is later authorized. Failure to finish the frozen comparisons makes its gate unevaluable, not passed.

Prior constraints: [Adam](https://arxiv.org/abs/1412.6980) uses historical adaptive moments; [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) balances task gradient magnitudes; [PCGrad](https://arxiv.org/abs/2001.06782) projects conflicting task gradients. Optimizer-aware balancing or projection is not itself novel. Any later contribution must demonstrate a counting-specific useful difference and compare against these ideas, alongside the already identified DM-Count/PML loss precedents.

The completed test's notification monitor is being retired after this verified result. No further training, diagnostic or automation is launched by this decision.
