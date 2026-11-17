# V15 completed instance-mass source experiment

Verified October 26, 2026 at 10:25 KST (client time). V15 completed all 56 source epochs / 25,592 optimizer steps and both the model and controller report completion. Docker exited 0 with OOMKilled=false. The prescribed gain gate failed. Retain V11.

The one owner-authorized exploratory run used 3,653 unchanged FSC source-training queries, fresh seed 2027, the same 470,359-parameter V11 architecture and frozen SAM2, and the same 204,568 training presentations. The intended difference replaced the 0.5 conserved-blur local L1 term with 0.5 joint exemplar/point-spacing instance-mass L1, retaining count, regional and marginal-CDF terms. The bounded scientific screen was negative; the CUDA engineering guard passed. Neither fact was treated as evidence of counting improvement.

## Final comparison on the same 1,286 reused native validation queries

| Method | MAE | RMSE | Bias | V15 minus method MAE |
| --- | ---: | ---: | ---: | ---: |
| V15 | 14.7626414955 | 68.6621541810 | -6.5059512396 | — |
| V11 | 12.4936851324 | 52.7660160755 | -1.1630913737 | +2.2689563631 |
| CountGD++ global | 8.8367029549 | 43.9662849763 | 1.5054432348 | +5.9259385406 |
| GeCo2 global | 9.4167962675 | 33.3107103116 | -4.2503888025 | +5.3458452280 |

Positive differences mean V15 is worse. V15 achieved neither MAE <9 nor MAE <8.

Paired bootstrap 95% intervals use 2,000 resamples, seed 2027, 1,225 exact-image groups and 29 linked category components. The local replay exactly reproduced all eight saved intervals (including the secondary native-unit comparison).

| V15 minus control | Image-group 95% CI | Category-component 95% CI |
| --- | --- | --- |
| V11 | [0.7596067436, 4.0340276703] | [0.3079914140, 4.9557438583] |
| CountGD++ global | [3.7351496341, 8.5093139488] | [2.7129824013, 10.5915368480] |
| GeCo2 global | [2.9542000826, 8.3776501444] | [2.3889940293, 9.3808551908] |

## Sparse, dense and spatial tradeoffs

| Native count stratum | Queries | V11 MAE | V15 MAE | V11 bias | V15 bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–24 | 603 | 2.560506 | 2.243539 | 1.791726 | 1.163302 |
| 25–74 | 419 | 6.301145 | 5.648844 | 1.909501 | -0.586832 |
| 75–149 | 149 | 13.238520 | 15.826970 | 4.969894 | -3.019345 |
| 150 and above | 115 | 86.175434 | 112.233207 | -35.797754 | -72.803126 |

The dense 115 queries contribute 67.99% of V15 absolute error, compared with 61.68% for V11. This configuration did not repair dense counting. Density alone does not establish occlusion or a causal failure mechanism.

| Protected/spatial measure | V11 | V15 |
| --- | ---: | ---: |
| GAME1 | 14.1883154692 | 16.5125669632 |
| GAME2 | 17.0851767380 | 19.4757506449 |
| Class-macro MAE | 11.9027882544 | 13.8681268454 |

Both required V11 MAE interval checks, RMSE, GAME1 and class-macro checks failed; only the sparse-MAE non-regression check passed. Sparse and 25–74-count gains do not compensate for the dense and aggregate regressions.

Training MAE is 4.3051043814; it is a fitting metric, not validation accuracy. Saved epoch durations sum to 44.4363 hours under shared GPU conditions, not exclusive timing. Equal exposure does not imply equal computation.

## Verification and preservation

Read-only verification checked all four result hashes, all 101 release files, all 15 frozen controller/job files, all 99 Python entries in the plan, and all 63 preserved V11 Python files. The plan matches the source rows, cache provenance, diagnostic and CUDA-guard hashes, control checkpoint, selected budget, parameter count and seed. All 56 finite training curves and timestamped saved-epoch logs agree with 25,592 prescribed steps and final resume-file metadata. No checkpoint was deserialized.

Every method covers exactly the same 1,286 unique native query IDs. Candidate/V11/CountGD native truth and groups match the preserved manifest; GeCo2 saved counts are joined by query ID to that same truth. Counts equal preserved point multiplicity; the source validation manifest SHA256 matches the hash pinned in its frozen feature provenance and its rows equal the preserved canonical manifest. Predictions are finite and failure-free. MAE, RMSE, bias, all four strata and class-macro MAE were independently recomputed. GAME1/2 were checked against hashed saved metrics; density maps were not regenerated.

The first local bootstrap replay used raw GeCo capture order and stopped at its image interval. The corrected replay follows the frozen evaluator’s canonical manifest ordering and exactly reproduces every interval. The initial local script is preserved. No scientific result, seed, grouping or remote file was changed.

Final checkpoint SHA256: `182b3564c9f8f026fc180f7fcba53a215d94f994c3be1ea5c56c09f2e3e04233`.

Prediction SHA256: `5c7768fbc9bb0b9ddc6492498f4b65f2283ae80944344f8d76a40ecd48ae30b9`.

Comparison SHA256: `984e566b2dfcd0abd48b2462a63eac8da779ea8c038a19d95ba302d0215949e9`.

This is one seed on reused source validation, with exact-image/category grouping rather than verified independent scenes. It establishes no external confirmation, foreground-sack accuracy, novel mechanism or SOTA claim. Preserve V11 and this negative V15 result. No repeat, repair, new fit, target tuning or reserved scoring was started.

The final-comparison monitoring purpose is complete; its deletion receipt is saved with the evidence. All older active/pending descriptions are historical.

Machine-readable result · Read-only verification · Exact interval replay
