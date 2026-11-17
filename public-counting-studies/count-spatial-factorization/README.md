# Four-arm count/spatial continuation result — October 28, 2026

**Frozen scientific gate: FAILED.** Verified at 2026-10-28T12:33:05.131454+09:00 (client Asia/Seoul).

Completed all 256 disposable scientific updates and four guard updates, with zero persistent model updates. Both Docker stages exited 0 without OOM; the one-shot controller completed. Independently verified 465 file identities, including all 110 release files, 321 selected feature files, native source/class maps and both starting checkpoints. All 14 readouts cover the same 65 fitting queries, including all seven extreme examples. All 8 arm/state endpoints have exactly 32 updates and Adam age 4,568, starting from age 4,536.

These are repeated fitting diagnostics. Update and readout exact groups are disjoint for this continuation, but both starting states previously trained on the readouts. No class-held, native validation, reserved or sack evaluation was performed. This result neither proves convergence nor rules out a different fresh-initialization outcome.

183 of 298 scientific checks failed. Engineering checks passed. A different arm cannot replace the prespecified primary after observing these results.

## All endpoint results

Lower is better. Original means a 32-update continuation under the unchanged original loss; baseline means the starting checkpoint before these updates. The primary is `explicit_count_spatial` (`C + S`).

### A0_original

| Endpoint | Overall MAE | Sparse MAE | Dense 150–899 MAE | Extreme 900+ MAE | Dense 150+ MAE | GAME1 / 2 / 3, all | Normalized spatial L1, all |
|---|---:|---:|---:|---:|---:|---|---:|
| baseline | 79.330583 | 1.919527 | 100.774459 | 389.920053 | 170.568223 | 81.658435 / 90.558330 / 102.139697 | 1.153975 |
| original | 70.740264 | 2.059392 | 80.170619 | 372.093506 | 150.634764 | 76.644908 / 87.081782 / 101.006340 | 1.178707 |
| count_only | 101.600656 | 1.701532 | 121.457695 | 539.209516 | 222.294342 | 104.103967 / 110.113616 / 120.810242 | 1.239963 |
| count_weight_control | 101.176398 | 1.767674 | 121.133696 | 535.994306 | 221.272464 | 103.657602 / 109.665260 / 120.311932 | 1.229430 |
| explicit_count_spatial | 75.150372 | 2.971078 | 79.011290 | 422.303955 | 161.875037 | 81.133187 / 90.226108 / 104.439069 | 1.167556 |
| matched_original | 70.740264 | 2.059392 | 80.170619 | 372.093506 | 150.634764 | 76.644908 / 87.081782 / 101.006340 | 1.178707 |
| matched_candidate | 71.434795 | 2.401223 | 78.304203 | 394.270874 | 154.572020 | 77.206414 / 86.735557 / 100.492076 | 1.143422 |

Primary minus unchanged continuation; negative is improvement:

| Group | MAE difference | MAE change | RMSE difference | GAME1 | GAME2 | GAME3 | Normalized spatial L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | +4.410109 | +6.234% | +19.655717 | +4.488279 | +3.144326 | +3.432730 | -0.011151 |
| sparse_0_24 | +0.911686 | +44.270% | +1.864234 | +0.436858 | -0.458964 | -0.765092 | -0.062450 |
| medium_25_74 | +0.364993 | +4.545% | +0.687243 | -1.684462 | -0.114443 | +0.560134 | +0.019951 |
| medium_75_149 | -10.199081 | -56.550% | -10.652890 | -9.322434 | -8.640669 | -5.513428 | +0.044120 |
| dense_150_plus | +11.240272 | +7.462% | +29.529727 | +12.268283 | +9.191342 | +9.169258 | +0.002068 |
| dense_150_899 | -1.159330 | -1.446% | +3.448631 | -2.291225 | -3.523883 | -4.451358 | +0.007676 |
| extreme_900_plus | +50.210449 | +13.494% | +66.493265 | +58.026736 | +49.153477 | +51.976908 | -0.015558 |

Extreme examples (absolute error; every example retained):

| Query | Native count | Baseline AE | Original AE | Primary AE | Primary − original |
|---|---:|---:|---:|---:|---:|
| 2360.jpg | 1283 | 293.094727 | 326.091614 | 479.922485 | +153.830872 |
| 7603.jpg | 1400 | 124.849365 | 203.432617 | 370.720581 | +167.287964 |
| 805.jpg | 1095 | 314.824951 | 316.700439 | 271.987183 | -44.713257 |
| 2347.jpg | 1912 | 937.386902 | 866.001709 | 928.044556 | +62.042847 |
| 2775.jpg | 1672 | 791.169800 | 750.810486 | 860.279968 | +109.469482 |
| 804.jpg | 1001 | 166.563110 | 93.826782 | 1.569702 | -92.257080 |
| 4128.jpg | 953 | 101.551514 | 47.790894 | 43.603210 | -4.187683 |

Actual update magnitudes and clipping:

| Arm | Parameter displacement L2 | Clipped fraction | Update seconds |
|---|---:|---:|---:|
| original | 0.92878194 | 1.0000 | 137.80 |
| count_only | 1.80657063 | 1.0000 | 136.55 |
| count_weight_control | 1.79719078 | 1.0000 | 137.07 |
| explicit_count_spatial | 1.32630781 | 1.0000 | 135.94 |

The equal-displacement endpoints shrink original and primary updates to their smaller common norm; they do not extrapolate or perform optimizer calls. Runtime is shared-GPU diagnostic timing, not exclusive throughput.

### A1_count_only

| Endpoint | Overall MAE | Sparse MAE | Dense 150–899 MAE | Extreme 900+ MAE | Dense 150+ MAE | GAME1 / 2 / 3, all | Normalized spatial L1, all |
|---|---:|---:|---:|---:|---:|---|---:|
| baseline | 138.042560 | 2.068491 | 113.443804 | 890.669591 | 301.050028 | 144.505897 / 153.300272 / 167.986733 | 1.672943 |
| original | 142.176770 | 1.632014 | 115.595406 | 927.747301 | 311.632070 | 148.384486 / 155.819563 / 171.162467 | 1.669563 |
| count_only | 151.707906 | 3.089081 | 139.017993 | 943.087878 | 333.103828 | 157.325098 / 166.105143 / 180.847937 | 1.682126 |
| count_weight_control | 151.721705 | 3.059544 | 139.120483 | 942.966291 | 333.152230 | 157.323830 / 166.080791 / 180.809385 | 1.681967 |
| explicit_count_spatial | 175.778380 | 13.390644 | 178.086079 | 958.503802 | 366.462771 | 180.237957 / 184.798923 / 191.372503 | 1.668258 |
| matched_original | 142.176770 | 1.632014 | 115.595406 | 927.747301 | 311.632070 | 148.384486 / 155.819563 / 171.162467 | 1.669563 |
| matched_candidate | 143.270272 | 8.351549 | 124.629769 | 874.483481 | 305.628941 | 148.583337 / 154.168047 / 165.628286 | 1.659995 |

Primary minus unchanged continuation; negative is improvement:

| Group | MAE difference | MAE change | RMSE difference | GAME1 | GAME2 | GAME3 | Normalized spatial L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | +33.601610 | +23.634% | +28.625696 | +31.853472 | +28.979360 | +20.210036 | -0.001305 |
| sparse_0_24 | +11.758630 | +720.498% | +12.636028 | +6.662071 | +2.090172 | -4.852801 | -0.029482 |
| medium_25_74 | +28.514447 | +324.124% | +25.766137 | +23.149516 | +15.569120 | +3.147231 | -0.045002 |
| medium_75_149 | +12.282874 | +88.212% | +13.404900 | +11.025666 | +10.621935 | +5.257487 | -0.082936 |
| dense_150_plus | +54.830701 | +17.595% | +42.191106 | +56.537417 | +55.945937 | +46.472108 | +0.050084 |
| dense_150_899 | +62.490673 | +54.060% | +63.798314 | +64.740441 | +64.958697 | +56.092561 | +0.004551 |
| extreme_900_plus | +30.756500 | +3.315% | +45.038465 | +30.756483 | +27.620117 | +16.236398 | +0.193188 |

Extreme examples (absolute error; every example retained):

| Query | Native count | Baseline AE | Original AE | Primary AE | Primary − original |
|---|---:|---:|---:|---:|---:|
| 2360.jpg | 1283 | 793.949402 | 845.436218 | 863.944244 | +18.508026 |
| 7603.jpg | 1400 | 850.186340 | 909.088257 | 1046.437988 | +137.349731 |
| 805.jpg | 1095 | 662.193542 | 696.176971 | 536.205811 | -159.971161 |
| 2347.jpg | 1912 | 1535.321228 | 1572.486969 | 1451.719910 | -120.767059 |
| 2775.jpg | 1672 | 1286.834686 | 1272.218811 | 1516.313843 | +244.095032 |
| 804.jpg | 1001 | 579.065002 | 617.235504 | 870.649628 | +253.414124 |
| 4128.jpg | 953 | 527.136932 | 581.588379 | 424.255188 | -157.333191 |

Actual update magnitudes and clipping:

| Arm | Parameter displacement L2 | Clipped fraction | Update seconds |
|---|---:|---:|---:|
| original | 0.77366640 | 1.0000 | 136.90 |
| count_only | 1.19349175 | 1.0000 | 136.87 |
| count_weight_control | 1.19320069 | 1.0000 | 136.91 |
| explicit_count_spatial | 2.73930285 | 1.0000 | 136.90 |

The equal-displacement endpoints shrink original and primary updates to their smaller common norm; they do not extrapolate or perform optimizer calls. Runtime is shared-GPU diagnostic timing, not exclusive throughput.

## Full gate accounting

| Starting state | Candidate | Reference | Check type | Failed / total |
|---|---|---|---|---:|
| A0_original | explicit_count_spatial | baseline | gain_1percent | 4/6 |
| A0_original | explicit_count_spatial | original | gain_1percent | 4/6 |
| A0_original | explicit_count_spatial | count_only | gain_1percent | 0/3 |
| A0_original | explicit_count_spatial | count_weight_control | gain_1percent | 0/3 |
| A0_original | explicit_count_spatial | baseline | protection | 20/37 |
| A0_original | explicit_count_spatial | baseline | extreme_individual_protection | 3/7 |
| A0_original | explicit_count_spatial | original | protection | 22/37 |
| A0_original | explicit_count_spatial | original | extreme_individual_protection | 4/7 |
| A0_original | matched_candidate | matched_original | gain_1percent | 2/6 |
| A0_original | matched_candidate | matched_original | protection | 12/37 |
| A1_count_only | explicit_count_spatial | baseline | gain_1percent | 6/6 |
| A1_count_only | explicit_count_spatial | original | gain_1percent | 6/6 |
| A1_count_only | explicit_count_spatial | count_only | gain_1percent | 3/3 |
| A1_count_only | explicit_count_spatial | count_weight_control | gain_1percent | 3/3 |
| A1_count_only | explicit_count_spatial | baseline | protection | 32/37 |
| A1_count_only | explicit_count_spatial | baseline | extreme_individual_protection | 4/7 |
| A1_count_only | explicit_count_spatial | original | protection | 32/37 |
| A1_count_only | explicit_count_spatial | original | extreme_individual_protection | 4/7 |
| A1_count_only | matched_candidate | matched_original | gain_1percent | 4/6 |
| A1_count_only | matched_candidate | matched_original | protection | 18/37 |

Every signed metric and individual gate check is retained in analysis.json. The result archive retains raw endpoints, native identities, timing, admissions, guard receipts and the independent verification ledger.

No full fit, new seed, retry or changed primary follows automatically. Retain V11 until controlled evidence supports replacement. See the separate decision receipt and pretrained strategy review.
