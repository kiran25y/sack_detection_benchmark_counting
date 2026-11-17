# Exemplar clarification feasibility: verified result, October 28, 2026

Verified at 2026-10-28 20:03 KST. **The prespecified combined headroom gate FAILED.** All 680 forwards completed; controller and all four containers exited zero without OOM. There were zero optimizer or persistent steps. No policy pilot or full fit follows this result.

This cohort contains 128 stratified FSC training images, potentially seen by the pretrained counters. These are feasibility scores, not population MAE, unseen accuracy, or the existing 1,286-query validation benchmark. Oracle selection reads ground truth and cannot be deployed.

## Overall MAE and controls

All request policies retain all 128 images in scoring. The primary oracle may request at most 32 additional native exemplars. Random and confidence controls request exactly 32; random values are exact expected aggregate losses.

| Comparison | Always two | Always three | Random 25% | Confidence 25% | Oracle up to 25% | Helped / hurt / unchanged |
|---|---:|---:|---:|---:|---:|---:|
| CountGD++ native global | 31.828125 | 32.648438 | 32.033203 | 32.515625 | 31.531250 | 18 / 30 / 80 |
| GeCo2 native | 21.304688 | 18.335938 | 20.562500 | 18.156250 | 16.312500 | 49 / 30 / 49 |
| GeCo2 fixed pair scale | 21.304688 | 18.359375 | 20.568359 | 18.171875 | 16.406250 | 41 / 34 / 53 |

The oracle used 18 CountGD++ requests and 32 requests for each GeCo2 comparison. The observed helpful-case counts above describe always-three outcomes, not generalization or a learned selector.

## Frozen primary gate

| Comparison | Overall MAE gain | Dense 150+ gain | Extreme 900+ gain | Protection gates | Decision |
|---|---:|---:|---:|---|---|
| CountGD++ native global | 0.933% | 0.796% | 0.000% | All passed within frozen margins | FAIL component |
| GeCo2 native | 23.432% | 23.973% | 44.769% | All passed within frozen margins | PASS component |
| GeCo2 fixed pair scale | 22.992% | 23.578% | 44.769% | All passed within frozen margins | PASS component |

CountGD++ failed exactly three conditions: overall practical gain, combined-dense gain and extreme gain. Overall improvement was 0.296875 objects, below the required 1.59140625 (5% of 31.828125); dense improvement was 0.553571 versus a required 1.391429; extreme improvement was zero versus a required 8.617143. At least eight helped and all oracle protections passed. Both GeCo2 comparisons passed all their component conditions. The protocol required all three comparisons, so the combined decision remains **fail**.

The improvement percentages are from truth-based selection. Protection means within max(0.1 objects, 1% of the baseline metric); it does not mean every metric or every image improved.

| Oracle comparison | Dense MAE before → after | Extreme MAE before → after | Sparse MAE before → after | Sparse GAME1/2/3 changes |
|---|---:|---:|---:|---|
| CountGD++ native global | 69.571429 → 69.017857 | 430.857143 → 430.857143 | 1.843750 → 1.812500 | -0.03125, -0.03125, -0.03125 |
| GeCo2 native | 45.214286 → 34.375000 | 92.857143 → 51.285714 | 2.093750 → 1.781250 | -0.12500, +0.00000, -0.18750 |
| GeCo2 fixed pair scale | 45.214286 → 34.553571 | 92.857143 → 51.285714 | 2.093750 → 1.875000 | -0.09375, -0.03125, +0.03125 |

## Material costs and what the controls establish

- CountGD++ always-three worsened overall MAE by +0.8203125, dense MAE by +1.714286 and sparse MAE by +0.25. Sparse RMSE rose from 4.369640 to 5.071366, with sparse GAME1/2/3 increases of +0.3125/+0.375/+0.4375. The confidence policy also worsened overall MAE by +0.6875. More examples are not uniformly beneficial.
- GeCo2 native always-three improved overall MAE by 2.96875, but medium-25–74 RMSE rose from 3.702477 to 4.587120. Its fixed-scale counterpart improved overall MAE by 2.9453125. The close aggregate gains with the scale fixed show opportunity beyond changing resize; they do not prove semantic ambiguity is the sole cause.
- At 25%, GeCo2 confidence selection improved overall MAE to 18.15625 (native) / 18.171875 (fixed scale), but medium-75–149 MAE rose 4.3125 → 4.625 and RMSE 6.571720 → 6.809919. Both increases exceed the 0.1 protection allowance. A useful aggregate confidence heuristic is not yet a protected policy.
- GeCo2 oracle dense GAME1/2/3 changes were −7.339286/−6.910714/−4.982143 (native) and −7.160714/−6.589286/−4.625 (fixed scale). Extreme changes were −17/−20.428571/−6.428571 for both. Fixed-scale sparse GAME3 increased +0.03125, within its 0.1 allowance. Native sparse GAME2 was unchanged.
- Always-three extreme GeCo2 GAME3 worsened +1.285714 while extreme MAE improved; this increase is below the frozen 1% allowance (2.28). Global error and localization error are distinct.

Paired image-resampling 95% intervals for always-three minus always-two MAE:
- CountGD++ native global: [0.007812, 2.172070].
- GeCo2 native: [-7.734375, 0.976758].
- GeCo2 fixed pair scale: [-7.664258, 1.062500].

Both GeCo2 intervals span zero. These intervals resample this stratified source-training bank; image hashes do not establish scene independence. They are not a validation or novelty claim.

## All seven extreme examples

CountGD++ predicted exactly 900 with two and three exemplars on every extreme. Its measured extreme MAE equals the known configuration floor, 430.857142857, contributing 23.5625 to the whole-bank MAE. An extra exemplar cannot remove the fixed global query cap. Cropping is an existing author strategy and was not added here.

Native and fixed-scale GeCo2 produced identical counts on these seven examples; both retain all seven in scoring.

| Query | Native truth | CountGD++ two / three | GeCo2 two / three | GeCo2 absolute-error gain |
|---|---:|---:|---:|---:|
| 2360.jpg | 1283 | 900 / 900 | 1244 / 1248 | +4 |
| 2775.jpg | 1672 | 900 / 900 | 1515 / 1598 | +83 |
| 805.jpg | 1095 | 900 / 900 | 1018 / 1205 | -33 |
| 7603.jpg | 1400 | 900 / 900 | 1397 / 1304 | -93 |
| 804.jpg | 1001 | 900 / 900 | 1133 / 1130 | +3 |
| 4128.jpg | 953 | 900 / 900 | 911 / 913 | +2 |
| 2347.jpg | 1912 | 900 / 900 | 1712 / 1911 | +199 |

## Secondary 50% request rate

This rate cannot rescue the primary gate. CountGD++ already uses all 18 beneficial cases at 25%, so its oracle gain does not increase.

| Comparison | Oracle MAE (actual requests) | Random expected MAE | Confidence MAE |
|---|---:|---:|---:|
| CountGD++ native global | 31.531250 (18) | 32.238281 | 32.601562 |
| GeCo2 native | 16.179688 (49) | 19.820312 | 18.390625 |
| GeCo2 fixed pair scale | 16.335938 (41) | 19.832031 | 18.500000 |

## Verification and decision

Independent stdlib recomputation checked all 1,008 policy/group MAE, RMSE, bias and GAME1/2/3 aggregates, matching within 5.69e-14. Exact query/variant/repeat coverage, all 128 unique image hashes, finite outputs, zero steps, four successful Docker exits, controller exit, five remote release files, seven input/controller/analyzer hashes and the source manifest identity passed. All four worker reports preserved finite and unmodified parameter checks. The original frozen analyzer was used unchanged; its shared scoring source also matches the release.

Retain V11 and all previous negative results. Do not train a clarification policy or another counter under this failed gate. Preserve the GeCo2 oracle opportunity as model-specific feasibility evidence without changing the frozen cross-model decision. Its predictability before seeing the extra exemplar, transfer to unconsumed groups/domains, and a distinct mechanism remain untested. No new inference, optimization, cropping, coefficient search or favorable-arm switch was launched.

Prior work already covers exemplar selection, interaction, information-acquisition utility and deferral; see the prior-work audit. A generic benefit regressor or a two-versus-three curve alone does not establish novelty. This study simulates disclosure of existing native boxes; human effort was not measured. Paper acceptance is not established by these findings.

Evidence: full analysis, independent verification, result receipt, decision receipt. The original archive and all stage outputs remain preserved.

The one-experiment completion monitor `clarification-feasibility-result` was deleted after verification. No new automation or execution was started.
