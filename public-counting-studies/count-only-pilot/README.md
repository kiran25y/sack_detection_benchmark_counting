# Count-only matched pilot: verified result, October 28, 2026

Verified at 2026-10-28 11:31 KST. **Completed; count-learning promotion gate FAILED.** Both arms completed exactly 14 epochs / 4,536 updates each (9,072 total) and scored every one of the same 1,066 class-held development queries. The Docker container exited 0 without OOM; the one-shot controller completed and exited. No 56-epoch full refit started.

## Matched comparison

A0 retains the original count + regional + CDF + conserved-blur objective. A1 uses only the original count L1. Both use the same 470,359-parameter head, frozen SAM2 cache, 2,587 fitting rows, seed 2027 initial state, all 14 batch orders, AdamW, clipping and BF16/FP32 recipe.

| Metric, all 1,066 queries | Original A0 | Count-only A1 | A1 minus A0 |
|---|---:|---:|---:|
| MAE | 11.272708 | 13.131834 | +1.859126 |
| RMSE | 30.678777 | 37.163563 | +6.484787 |
| bias | 1.715785 | -8.149718 | -9.865502 |
| Class-macro MAE | 20.932737 | 19.933552 | -0.999186 |
| GAME1 | 13.407497 | 16.917158 | +3.509662 |
| GAME2 | 16.176970 | 22.448456 | +6.271486 |
| GAME3 | 21.857393 | 32.459657 | +10.602264 |
| Normalized spatial discrepancy | 1.199658 | 1.834369 | +0.634711 |

## Count strata

| Native count range | Queries | A0 MAE | A1 MAE | A1 minus A0 |
|---|---:|---:|---:|---:|
| Sparse: 0–24 | 487 | 3.557601 | 3.115282 | -0.442319 |
| Medium: 25–74 | 367 | 7.730211 | 9.782354 | +2.052144 |
| Medium: 75–149 | 156 | 21.745188 | 24.631886 | +2.886698 |
| Dense: 150+ | 56 | 72.409251 | 90.155253 | +17.746002 |
| Dense: 150–899 | 56 | 72.409251 | 90.155253 | +17.746002 |

There are **zero held-class examples with count >=900** in this unchanged partition; extreme-count efficacy is therefore unevaluable here. Dense 150+ and 150–899 are the same 56 queries. No examples were moved between partitions.

## Frozen gate

| Requirement | Observed | Result |
|---|---|---|
| Overall MAE improves at least 5% | Worsened 16.49% | FAIL |
| Dense MAE improves at least 10% | Worsened 24.51% | FAIL |
| 150–899 MAE improves | 72.409251 to 90.155253 | FAIL |
| Overall RMSE increase <=2% | Increased 21.14% | FAIL |
| Sparse MAE increase <=0.2 | Decreased 0.442319 | PASS |

## Interpretation and tradeoffs

Count-only improved sparse MAE and overall class-macro MAE, but overall and both medium-stratum MAEs worsened. Dense undercount bias shifted from -18.531935 to -84.225141. GAME1/2/3 and normalized spatial discrepancy worsened overall and in every populated count stratum, including sparse images. Medium 75–149 RMSE improved despite worse MAE; the complete metrics remain in the analysis receipt.

This result does not support removing the original spatial auxiliaries as the dense-count repair. It is a fixed 14-epoch, single-seed development evidence; it does not prove eventual convergence behavior or identify which individual auxiliary term helps. A routed spatial branch whose count branch exactly follows count-only cannot recover its count MAE solely by changing spatial distribution. Retain V11, preserve this negative result, and review saved errors before proposing a new bounded mechanism. No additional fit, sweep, seed or full refit was launched.

## Verification and limits

Independent standard-library analysis verified unique full coverage, unchanged native counts/classes/exact groups, disjoint fitting/held classes and exact groups, finite nonnegative predictions, every curve/step count, identical initialization and all 14 batch orders. All 105 release files, 17 result files (including both checkpoint files and resume files) and 2 source/class identities matched: 124 file hashes. Plan/protocol/guard identities and controller training exit matched. MAE/RMSE/bias/class-macro were independently recomputed from saved predictions; GAME and normalized-discrepancy summaries were independently aggregated from saved per-image values. No model was loaded and no GPU inference or optimization was used for verification.

These 1,066 reused class-held development scores must not be compared directly with the historical 1,286-query V11/CountGD++/GeCo2 validation leaderboard. There is no new native-validation, independent-domain, foreground-sack or novelty result.

Runtime was about 12.72 hours for the paired pipeline on a shared GPU. The host finish timestamp corresponds approximately to October 28, 01:42 KST after its known approximately eight-hour clock offset; live verification occurred at the client time above.

## Evidence

- Complete independent analysis
- Verified result receipt
- Hash and identity verification
- Preserved output archive
