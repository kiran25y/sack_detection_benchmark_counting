# Pretrained exemplar-part bounded test result — October 28, 2026

**Frozen scientific gate: FAILED.** Verified at 2026-10-28T13:18:26.489489+09:00 (client Asia/Seoul).

Completed all 128 disposable scientific Adam updates and 10 native guard updates; zero persistent model updates. Both native Docker stages exited 0 without OOM and the one-shot controller completed. Independently verified 461 file identities, including all 115 release files, all 321 selected feature files, the source/class maps, feature index and the authentic pretrained checkpoint. All nine readouts cover the same 65 queries and all seven extreme examples. Four arms each completed exactly 32 steps: original parameter Adam age 4,536 to 4,568; 561 added parameters age 0 to 32.

The original spatial objective was retained in every arm. Every arm has 470,920 parameters, with the extra 561 dormant in original continuation and active in the three capacity-matched adapters. Initial native predictions/base gradients, finite gradients and restored optimizer/RNG state passed. CUDA poststep recovery is numerical, not necessarily bitwise. No full training, saved trained checkpoint, seed repeat, coefficient search, recapture or internal/native/target/reserved validation was performed.

These are fitting readouts previously seen by the checkpoint, reused in development. New update/readout exact groups are disjoint, but this is not unseen validation or proof of independent scenes. A negative 32-update late-state screen does not prove failure from fresh initialization or over longer training.

216 of 303 prespecified scientific checks failed. All engineering prerequisites passed. The ordered-parts arm is the only prespecified primary; no other arm is promoted after inspecting results.

## Complete endpoint comparison

Lower is better. Baseline is the pretrained starting checkpoint. Every active endpoint uses the same 32 batches and unchanged original loss. Matched endpoints shrink all four full displacements to their smallest norm without additional optimization.

| Endpoint | Overall MAE | Sparse MAE | Dense 150–899 MAE | Extreme 900+ MAE | Dense 150+ MAE | GAME1 / 2 / 3, all | Normalized blur, all |
|---|---:|---:|---:|---:|---:|---|---:|
| baseline | 79.330583 | 1.919527 | 100.774459 | 389.920053 | 170.568223 | 81.658435 / 90.558330 / 102.139697 | 1.153975 |
| original | 71.123737 | 2.037003 | 80.712765 | 374.499939 | 151.626910 | 77.046382 / 87.269905 / 101.099057 | 1.178756 |
| summary_residual | 73.780878 | 1.981422 | 80.954462 | 398.859000 | 157.690040 | 79.838273 / 89.776265 / 103.105178 | 1.177323 |
| ordered_parts | 73.990709 | 1.977936 | 81.176488 | 399.961426 | 158.124576 | 80.010158 / 89.915671 / 103.251763 | 1.177319 |
| permuted_parts | 73.767628 | 1.973541 | 81.053585 | 398.026786 | 157.564358 | 79.758067 / 89.732573 / 103.116808 | 1.177798 |
| matched_original | 71.095709 | 2.014812 | 81.153396 | 372.766471 | 151.542759 | 76.901596 / 87.189173 / 100.906203 | 1.176440 |
| matched_summary_residual | 73.749579 | 1.977498 | 80.980028 | 398.588344 | 157.644104 | 79.819599 / 89.749188 / 103.073899 | 1.177183 |
| matched_ordered_parts | 73.990709 | 1.977936 | 81.176488 | 399.961426 | 158.124576 | 80.010158 / 89.915671 / 103.251763 | 1.177319 |
| matched_permuted_parts | 73.753932 | 1.977116 | 81.050992 | 397.950038 | 157.543865 | 79.740145 / 89.714483 / 103.092550 | 1.177726 |

## Signed differences against unchanged continuation

Ordered parts minus original; negative means improvement.

| Group | MAE difference | MAE change | RMSE difference | GAME1 | GAME2 | GAME3 | Normalized blur |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | +2.866971 | +4.031% | +7.869086 | +2.963777 | +2.645766 | +2.152706 | -0.001437 |
| sparse_0_24 | -0.059068 | -2.900% | +0.056456 | -0.032048 | -0.065418 | -0.160500 | -0.005324 |
| medium_25_74 | +0.295343 | +3.766% | +0.182408 | -0.514039 | +0.058050 | +0.226361 | -0.003435 |
| medium_75_149 | -0.641877 | -3.610% | -0.715007 | -0.644034 | -0.526930 | -0.562031 | +0.000428 |
| dense_150_plus | +6.497666 | +4.285% | +11.797252 | +6.975554 | +6.064284 | +4.973948 | +0.001547 |
| dense_150_899 | +0.463723 | +0.575% | +0.631203 | -0.286270 | -0.017650 | +0.221953 | -0.002323 |
| extreme_900_plus | +25.461487 | +6.799% | +27.543936 | +29.798429 | +25.178933 | +19.908789 | +0.013709 |

## Dense gains versus every information and capacity control

| Candidate | Reference | 150+ change | 150–899 change | 900+ change |
|---|---|---:|---:|---:|
| ordered_parts | baseline | -7.295% | -19.447% | +2.575% |
| ordered_parts | original | +4.285% | +0.575% | +6.799% |
| ordered_parts | summary_residual | +0.276% | +0.274% | +0.276% |
| ordered_parts | permuted_parts | +0.356% | +0.152% | +0.486% |
| matched_ordered_parts | matched_original | +4.343% | +0.028% | +7.295% |
| matched_ordered_parts | matched_summary_residual | +0.305% | +0.243% | +0.344% |
| matched_ordered_parts | matched_permuted_parts | +0.369% | +0.155% | +0.505% |

## Every extreme example

| Query | Native count | Baseline AE | Original AE | Summary AE | Ordered AE | Permuted AE | Ordered − original |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2360.jpg | 1283 | 293.094727 | 331.436768 | 361.461182 | 363.872437 | 361.208008 | +32.435669 |
| 7603.jpg | 1400 | 124.849365 | 208.101929 | 255.486328 | 258.274170 | 253.757568 | +50.172241 |
| 805.jpg | 1095 | 314.824951 | 320.858398 | 365.066040 | 365.193542 | 362.128540 | +44.335144 |
| 2347.jpg | 1912 | 937.386902 | 871.750244 | 908.095825 | 909.950806 | 906.179565 | +38.200562 |
| 2775.jpg | 1672 | 791.169800 | 753.944641 | 783.385864 | 784.582886 | 782.204956 | +30.638245 |
| 804.jpg | 1001 | 166.563110 | 86.289429 | 42.500854 | 40.361328 | 45.304443 | -45.928101 |
| 4128.jpg | 953 | 101.551514 | 49.118164 | 76.016907 | 77.494812 | 75.404419 | +28.376648 |

## Compute and displacement

| Arm | Full displacement L2 | Base displacement L2 | Clipped fraction | Update seconds | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| original | 0.92525317 | 0.92525317 | 1.0000 | 160.82 | 5.644 |
| summary_residual | 0.88346057 | 0.87980693 | 1.0000 | 160.42 | 5.642 |
| ordered_parts | 0.88020901 | 0.87676557 | 1.0000 | 160.78 | 5.642 |
| permuted_parts | 0.88192230 | 0.87827860 | 1.0000 | 162.44 | 5.643 |

Measured runtime is shared-GPU diagnostic timing. It does not measure exclusive throughput or other-job slowdown. The fixed permutation retains all part information and could be learned around. All active adapters use identical shapes/operations and initial parameters.

## Complete gate accounting

| Candidate | Reference | Check type | Failed / total |
|---|---|---|---:|
| ordered_parts | baseline | gain_1percent | 4/6 |
| ordered_parts | original | gain_1percent | 6/6 |
| ordered_parts | summary_residual | gain_1percent | 6/6 |
| ordered_parts | permuted_parts | gain_1percent | 3/3 |
| ordered_parts | baseline | protection | 22/37 |
| ordered_parts | baseline | extreme_individual_protection | 3/7 |
| ordered_parts | original | protection | 21/37 |
| ordered_parts | original | extreme_individual_protection | 6/7 |
| ordered_parts | summary_residual | protection | 26/37 |
| ordered_parts | summary_residual | extreme_individual_protection | 6/7 |
| matched_ordered_parts | matched_original | gain_1percent | 6/6 |
| matched_ordered_parts | matched_original | protection | 23/37 |
| matched_ordered_parts | matched_original | extreme_individual_protection | 6/7 |
| matched_ordered_parts | matched_summary_residual | gain_1percent | 6/6 |
| matched_ordered_parts | matched_summary_residual | protection | 30/37 |
| matched_ordered_parts | matched_summary_residual | extreme_individual_protection | 6/7 |
| matched_ordered_parts | matched_permuted_parts | gain_1percent | 6/6 |
| matched_ordered_parts | matched_permuted_parts | protection | 24/37 |
| matched_ordered_parts | matched_permuted_parts | extreme_individual_protection | 6/7 |

All signed checks, per-stratum MAE/RMSE/bias/class-macro/GAME and normalized discrepancy are retained in analysis.json. The completed archive retains every native endpoint, update log, resource sample, exit receipt and verified identity.

A practical threshold miss is insufficient evidence for promotion, not proof of no mathematical effect. Retain V11 pending stronger controlled evidence. No full training follows automatically. Ordered matching information, additional adapters and pretrained features have close prior work; this test cannot establish novelty.
