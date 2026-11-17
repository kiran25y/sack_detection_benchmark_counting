> October 26, 18:00 KST verified: this diagnostic was STOPPED by its unchanged 32-GiB host-memory guard at 28.1783 GiB available. Docker exited 137, OOMKilled=false. Ten disposable calls, zero complete saved batch comparisons, no final report. The scientific gate is UNEVALUABLE, not failed or passed. Old evaluating status is stale after shutdown. No restart/full fit; outputs preserved. The completion monitor was deleted. See `study_protocols/optimizer_interference_stop_20261026.json`.

# Bounded optimizer-interference diagnostic — October 26, 2026

## Current status: launched, gate pending

Owner authorized this one bounded test, including fair controls, and requested the result before any full training. The controller PID 96753 launched `source-optimizer-interference-v1` after two memory-admission checks (51.51 and 58.84 GiB available, 30 seconds apart). First verified phase was native replay, zero disposable calls at that snapshot; creation alone is not completed scientific evidence. No full training was launched.

Local bundle: `artifacts/optimizer_interference_v1_20261026`. Remote output: `research-direction-20261012/optimizer_interference_v1_20261026`. Frozen protocol SHA256: 99962a101a3e3dacea7f692b8da0f5df78c4bb62ffb2ee006c5d94633b61275c. Container ID: 2fba14a57ce04a461ed4de6993693e1a0a56ca6273e130a8c59122ac01ec08fd. Launch receipt: `study_protocols/optimizer_interference_launch_20261026.json`.

## Fixed comparison

Both authentic V11/V15 final states, original batches 0/2/4 (first sparse/medium/dense batch), only original V11 and anchored-candidate objectives. Preserve weights, second moments, optimizer age 25,592, learning rate, weight decay, architecture, inputs and native labels. Cross preserved/zero first moments with clip1/unclipped gradients. Reset before every proposal. Eight factorial calls, one zero-current-gradient call and one exact primary replay per context: six contexts and at most 60 disposable checkpoint Adam calls. Tiny synthetic optimizer checks use separate tensors and are not scientific model calls. No checkpoint writes or persistent steps.

All 65 fixed fitting images are evaluated for every raw proposal and its update-norm-matched counterpart. Roles are 8 update, 40 disjoint original fitting spillover, 17 supplemental; all seven extreme examples retained. Repeated evaluations are dependent. No class-held, native validation, sack or reserved scoring. Native replay and original preserved-momentum/clip1 proposals must reproduce saved prior results. First-moment intervention is counterfactual and is not fresh Adam or a production optimizer recommendation.

## Prespecified decision

The joint removal of both suspected factors, with candidate and primary steps matched to the original primary displacement norm, is fixed as the main diagnostic contrast. Other factorial conditions are reported and cannot replace it posthoc. This concretizes the earlier proposed factorial test before running it; it does not require the already-failed original setting to pass again.

Every applicable dense role and supplemental extreme cell in every context must improve by at least 1% over both the joint-intervention primary and the original primary, and beat the matched null beyond numerical tolerance. The norm-controlled candidate-minus-primary effect must improve over its original keep+clip counterpart. Protect sparse/medium MAE and GAME1/2/3 across all roles/strata against both primary controls. Both normalized spatial discrepancies must improve in all/dense/extreme and remain protected in other strata. Tolerance is unchanged: 1e-4 + 1e-6 times absolute comparator cell mass, summed for GAME. Normalized spatial tolerance is 1e-6. No averaging away a failed context.

Record full raw/matched factorial main effects and interaction, current primary/auxiliary directions before/after clipping, actual Adam directions and named parameter-group contributions. Analytic mean-count and squared normalized-spatial derivatives receive central-difference checks at both fixed radii .01/.005; report errors and possible nonlinear/FP32 effects, not a line search. Actual finite proposal metrics, not linear predictions, determine the scientific gate.

## Engineering and resources

Synthetic manual Adam matched all four interventions exactly; first-moment isolation, zero-gradient versus absent-gradient semantics, exact replay and norm matching passed. Copied objective checks passed with finite-difference max error 1.62e-9 and V15 gradient parity max difference 1.19e-7. Local analyzer threshold/boundary checks passed. These are engineering checks, not a gain.

Two CPUs, 16 GiB memory, no extra swap, no GPU/network, read-only research mount. Admission remains 48 GiB on two checks; host stop floor 32 GiB, disk floor 100 GiB, four-hour bound, stops only this container, no retry. Estimated order of runtime is roughly three hours based on the preceding diagnostic's similar number of proposal evaluations, conditional on runtime load; not a deadline. The four-hour guard is a cap, not a promise of completion.

On completion use exclusive collector `collect.py --name result_outputs.zip`, then local read-only `analyze_results.py --archive <absolute archive>`. Reuse existing valid outputs rather than rerun. Require six contexts/60 calls and independent release/input/checkpoint/result identities, successful Docker/controller exit, finite checks and all metrics before reporting pass/fail. Incomplete runs are unevaluable. Retain V11 unless future evidence justifies replacement. No automatic trajectory or full training follows even a pass.

Adam, GradNorm and PCGrad already constrain optimizer-balancing/projection novelty. This diagnostic can distinguish an optimizer-state explanation at these late states; it cannot establish learning from scratch, unseen accuracy or novelty.
