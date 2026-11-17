# Frozen-base residual result — October 28, 2026

**Verified FAILED gate at 2026-10-28 15:03:35 KST.** 139/300 scientific checks failed; all engineering prerequisites passed. Checklist checks are correlated, not independent statistical tests.

Completed64 scientific updates and8 guard updates; both containers and the controller exited0 without OOM. All120 release hashes,321 selected feature files and the source/class/checkpoint/historical identities were verified. Ten CPU tests passed. No persistent model update or full fit was made. The frozen original parameters and all their Adam states were bitwise unchanged after every scientific step.

## All endpoints

### all

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 79.330583 | 195.228375 | -43.972553 | 81.658435 | 90.558330 | 102.139697 | 1.153975 | 55.620283 |
| original | 71.123737 | 185.688208 | -50.443225 | 77.046382 | 87.269905 | 101.099057 | 1.178756 | 47.072490 |
| summary_residual | 73.780878 | 193.162168 | -55.487278 | 79.838273 | 89.776265 | 103.105178 | 1.177323 | 48.993905 |
| ordered_parts | 73.990709 | 193.557295 | -55.669204 | 80.010158 | 89.915671 | 103.251763 | 1.177319 | 49.174170 |
| frozen_summary | 80.091548 | 200.232011 | -50.181225 | 83.538342 | 92.046751 | 102.819546 | 1.153505 | 55.431957 |
| frozen_ordered | 79.973525 | 199.613217 | -49.501073 | 83.243953 | 91.830435 | 102.675135 | 1.153403 | 55.374793 |
| matched_frozen_summary | 80.017263 | 199.696814 | -49.514368 | 83.283043 | 91.867204 | 102.717320 | 1.153431 | 55.458707 |
| matched_frozen_ordered | 79.973525 | 199.613217 | -49.501073 | 83.243953 | 91.830435 | 102.675135 | 1.153403 | 55.374793 |

### sparse_0_24

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.919527 | 3.117169 | -0.272975 | 3.504993 | 4.639897 | 6.575392 | 1.448242 | 1.975363 |
| original | 2.037003 | 2.960097 | 0.191929 | 3.783200 | 5.397720 | 7.379129 | 1.478673 | 2.231975 |
| summary_residual | 1.981422 | 3.033087 | 0.041531 | 3.751797 | 5.330413 | 7.213961 | 1.472962 | 2.153009 |
| ordered_parts | 1.977936 | 3.016553 | 0.048640 | 3.751152 | 5.332302 | 7.218628 | 1.473349 | 2.148473 |
| frozen_summary | 1.901543 | 3.473919 | -0.649427 | 3.478074 | 4.603358 | 6.527595 | 1.442333 | 1.994467 |
| frozen_ordered | 1.917643 | 3.445518 | -0.597744 | 3.485647 | 4.606038 | 6.519597 | 1.442700 | 2.006816 |
| matched_frozen_summary | 1.923153 | 3.453352 | -0.604848 | 3.484485 | 4.608080 | 6.524876 | 1.442714 | 2.014682 |
| matched_frozen_ordered | 1.917643 | 3.445518 | -0.597744 | 3.485647 | 4.606038 | 6.519597 | 1.442700 | 2.006816 |

### medium_25_74

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 8.619029 | 15.219950 | -2.635805 | 12.317182 | 14.913470 | 20.422186 | 1.373673 | 9.106602 |
| original | 7.842904 | 10.345549 | 4.816789 | 14.765482 | 18.456188 | 24.163762 | 1.374319 | 7.698327 |
| summary_residual | 8.090481 | 10.556358 | 4.933010 | 14.372595 | 18.556722 | 24.359824 | 1.371272 | 8.070557 |
| ordered_parts | 8.138247 | 10.527957 | 5.028831 | 14.251443 | 18.514238 | 24.390124 | 1.370884 | 8.117713 |
| frozen_summary | 9.304629 | 17.579586 | -4.876329 | 12.028735 | 14.726300 | 19.832033 | 1.381095 | 9.946900 |
| frozen_ordered | 9.246536 | 17.379125 | -4.652131 | 12.075238 | 14.750467 | 19.898221 | 1.380470 | 9.878093 |
| matched_frozen_summary | 9.258403 | 17.390076 | -4.681315 | 12.079477 | 14.758559 | 19.897901 | 1.380604 | 9.889501 |
| matched_frozen_ordered | 9.246536 | 17.379125 | -4.652131 | 12.075238 | 14.750467 | 19.898221 | 1.380470 | 9.878093 |

### medium_75_149

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 14.238094 | 16.661762 | 14.238094 | 14.888323 | 18.826899 | 26.224969 | 0.889601 | 11.542811 |
| original | 17.782238 | 19.719245 | 17.782238 | 19.002224 | 22.898162 | 30.508464 | 0.919412 | 15.489787 |
| summary_residual | 17.035441 | 18.907135 | 17.035441 | 18.283518 | 22.316566 | 29.932718 | 0.920376 | 14.910947 |
| ordered_parts | 17.140361 | 19.004238 | 17.140361 | 18.358189 | 22.371231 | 29.946433 | 0.919840 | 15.002515 |
| frozen_summary | 13.227604 | 15.692014 | 13.227604 | 13.883884 | 18.330909 | 25.877343 | 0.886712 | 10.074569 |
| frozen_ordered | 13.283085 | 15.748989 | 13.283085 | 13.959707 | 18.367314 | 25.946170 | 0.886976 | 10.179009 |
| matched_frozen_summary | 13.281748 | 15.747586 | 13.281748 | 13.944157 | 18.390122 | 25.922580 | 0.886843 | 10.164700 |
| matched_frozen_ordered | 13.283085 | 15.748989 | 13.283085 | 13.959707 | 18.367314 | 25.946170 | 0.886976 | 10.179009 |

### dense_150_plus

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 170.568223 | 292.034551 | -100.407826 | 173.282643 | 190.737842 | 211.931062 | 0.929973 | 191.450102 |
| original | 151.626910 | 277.776179 | -118.534810 | 161.058022 | 180.780640 | 206.867852 | 0.958138 | 172.341566 |
| summary_residual | 157.690040 | 288.982297 | -129.622291 | 167.621543 | 186.530427 | 211.529752 | 0.959716 | 177.599819 |
| ordered_parts | 158.124576 | 289.573430 | -130.089708 | 168.033576 | 186.844924 | 211.841800 | 0.959685 | 178.042900 |
| frozen_summary | 172.258892 | 299.495355 | -113.082531 | 177.822073 | 194.266318 | 213.763249 | 0.931034 | 191.993516 |
| frozen_ordered | 171.991806 | 298.571739 | -111.682484 | 177.125288 | 193.763757 | 213.408021 | 0.930711 | 191.743695 |
| matched_frozen_summary | 172.082223 | 298.696741 | -111.697045 | 177.215461 | 193.837253 | 213.503923 | 0.930746 | 191.919727 |
| matched_frozen_ordered | 171.991806 | 298.571739 | -111.682484 | 177.125288 | 193.763757 | 213.408021 | 0.930711 | 191.743695 |

### dense_150_899

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 100.774459 | 182.150628 | -32.664356 | 100.854967 | 117.862573 | 135.362416 | 0.989372 | 127.707976 |
| original | 80.712765 | 170.997373 | -44.935854 | 84.956062 | 106.096979 | 129.312531 | 1.015955 | 112.578900 |
| summary_residual | 80.954462 | 171.450400 | -47.819780 | 84.498178 | 105.973660 | 129.441107 | 1.013972 | 112.359482 |
| ordered_parts | 81.176488 | 171.628576 | -47.890646 | 84.669792 | 106.079329 | 129.534484 | 1.013631 | 112.588679 |
| frozen_summary | 99.348114 | 183.554234 | -39.239972 | 100.067695 | 116.104391 | 133.386143 | 0.987387 | 127.616085 |
| frozen_ordered | 99.521073 | 183.393236 | -38.477389 | 100.168089 | 116.292842 | 133.557166 | 0.987273 | 127.602543 |
| matched_frozen_summary | 99.494366 | 183.402670 | -38.530521 | 100.142394 | 116.266613 | 133.564681 | 0.987437 | 127.605873 |
| matched_frozen_ordered | 99.521073 | 183.393236 | -38.477389 | 100.168089 | 116.292842 | 133.557166 | 0.987273 | 127.602543 |

### extreme_900_plus

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 389.920053 | 499.042865 | -313.315874 | 400.912480 | 419.774401 | 452.575378 | 0.743293 | 420.021708 |
| original | 374.499939 | 477.246017 | -349.845816 | 400.235611 | 415.500717 | 450.613148 | 0.776430 | 388.916720 |
| summary_residual | 398.859000 | 503.575359 | -386.715899 | 428.866406 | 439.708836 | 469.522638 | 0.789196 | 415.308272 |
| ordered_parts | 399.961426 | 504.789953 | -388.429618 | 430.034040 | 440.679650 | 470.521938 | 0.790140 | 416.536051 |
| frozen_summary | 401.407052 | 515.474692 | -345.159145 | 422.192974 | 439.918089 | 466.377010 | 0.753923 | 425.075651 |
| frozen_ordered | 399.756967 | 513.430991 | -341.755641 | 418.990773 | 437.243779 | 464.367852 | 0.752948 | 423.827502 |
| matched_frozen_summary | 400.215489 | 513.721532 | -341.648978 | 419.445101 | 437.630692 | 464.741540 | 0.752573 | 424.560311 |
| matched_frozen_ordered | 399.756967 | 513.430991 | -341.755641 | 418.990773 | 437.243779 | 464.367852 | 0.752948 | 423.827502 |

## Primary changes against every fixed control

| Reference | Dense150–899 MAE change | Extreme900+ MAE change | Dense150+ MAE change | Sparse MAE change | Sparse RMSE change | Overall GAME1/2/3 change |
|---|---:|---:|---:|---:|---:|---|
| baseline | -1.2438% | +2.5228% | +0.8346% | -0.0981% | +10.5336% | +1.9416% / +1.4047% / +0.5242% |
| original | +23.3028% | +6.7442% | +13.4309% | -5.8596% | +16.3988% | +8.0439% / +5.2258% / +1.5589% |
| summary_residual | +22.9346% | +0.2251% | +9.0695% | -3.2189% | +13.5977% | +4.2657% / +2.2881% / -0.4171% |
| ordered_parts | +22.5984% | -0.0511% | +8.7698% | -3.0483% | +14.2204% | +4.0417% / +2.1295% / -0.5585% |
| frozen_summary | +0.1741% | -0.4111% | -0.1550% | +0.8467% | -0.8175% | -0.3524% / -0.2350% / -0.1405% |

Positive errors are costs; negative errors are gains. The unchanged original continuation is the reference needed to distinguish ordinary additional training from the new mechanism. Historical joint controls are reused with verified identical inputs and initial state; they were not retrained. Updated parameter count differs by design between joint and frozen conditions.

## Every extreme example

| Query | Truth | Original AE | Joint ordered AE | Frozen summary AE | Frozen ordered AE | Frozen ordered prediction |
|---|---:|---:|---:|---:|---:|---:|
| 2360.jpg | 1283 | 331.436768 | 363.872437 | 316.255981 | 314.368530 | 968.631470 |
| 7603.jpg | 1400 | 208.101929 | 258.274170 | 163.615356 | 158.312988 | 1241.687012 |
| 805.jpg | 1095 | 320.858398 | 365.193542 | 360.023987 | 354.814148 | 740.185852 |
| 2347.jpg | 1912 | 871.750244 | 909.950806 | 963.134644 | 959.955688 | 952.044312 |
| 2775.jpg | 1672 | 753.944641 | 784.582886 | 809.951721 | 807.842773 | 864.157227 |
| 804.jpg | 1001 | 86.289429 | 40.361328 | 115.248047 | 121.068970 | 1122.068970 |
| 4128.jpg | 953 | 49.118164 | 77.494812 | 81.619629 | 81.935669 | 1034.935669 |

## Gate components

Failures by reference: {"baseline": 27, "original": 27, "summary_residual": 22, "ordered_parts": 22, "frozen_summary": 24, "matched_frozen_summary": 17}.

Failures by kind: {"gain_1percent": 26, "protection": 92, "extreme_individual_protection": 21}. Every failed component is retained in artifacts/frozen_residual_test_v1_20261028/analysis.json; no threshold, reference or primary was changed after execution.

## Capacity, optimizer and displacement controls

- frozen_summary: {"steps": 32, "seconds": 163.87306523200823, "parameter_displacement_norm": 0.09172030175262469, "base_displacement_norm": 0.0, "base_weights_optimizer_unchanged_every_step": true, "final_base_optimizer_age": 4536, "final_adapter_optimizer_age": 32, "exact_restored_start": true, "clip_fraction": 1.0, "peak_allocated_bytes": 6062724096, "peak_reserved_bytes": 6746537984}
- frozen_ordered: {"steps": 32, "seconds": 162.81347339198692, "parameter_displacement_norm": 0.08247044951158208, "base_displacement_norm": 0.0, "base_weights_optimizer_unchanged_every_step": true, "final_base_optimizer_age": 4536, "final_adapter_optimizer_age": 32, "exact_restored_start": true, "clip_fraction": 1.0, "peak_allocated_bytes": 6062056448, "peak_reserved_bytes": 6754926592}
- matched_frozen_summary: {"scale": 0.8991515284588789, "common_norm": 0.08247044951158208, "actual_norm": 0.0824704521033604}
- matched_frozen_ordered: {"scale": 1.0, "common_norm": 0.08247044951158208, "actual_norm": 0.08247044951158208}

Global clipping included original and adapter gradients before original gradients were masked. Thus the first adapter updates matched the joint controls exactly on the fixed native guard batch. The original470359 weights, moments and age4536 then remained fixed; only561 parameters advanced to age32. Complete new trajectories were finite. A regression against the same starting checkpoint in a frozen arm occurs without original-weight co-adaptation. A difference against joint original continuation also removes the benefit of original-weight learning and must not be treated as a pure causal decomposition. This test does not identify which residual activations cause an output change.

The new frozen endpoints were shrunk to their common smaller adapter-displacement norm. This comparison controls the two new arms only. Discarded historical joint endpoints cannot be reconstructed or rescaled from predictions. All old failed matched-control evidence remains preserved.

The signed summary/ordered by joint/frozen interaction for each metric is saved in the result receipt. It is descriptive on reused fitting images, not an independent causal effect estimate with uncertainty.

## Interpretation of the controlled failure

Both frozen adapters lowered60/65 predictions, including every dense and extreme query, against the starting checkpoint. The primary extreme MAE worsened2.523% and sparse RMSE10.534% even with the original weights and Adam state fixed. Its extreme improvement over joint ordered training was only0.051%, below the1% gate. See the decision and next diagnostic proposal for the common-offset hypothesis and its unresolved causal limits. No additional diagnostic or full training was launched.

## Decision and scope

Retain V11 and report the frozen gate before any full run. Do not select the summary arm, another seed, checkpoint, coefficient or endpoint as a new winner. This bounded screen does not rule out longer training or fresh-initialization learning. No external, native validation, class-held, reserved or sack accuracy was measured.

The proposed mechanism and controls were frozen in the protocol document. Frozen-base residual adaptation has prior precedent in [Houlsby et al.](https://proceedings.mlr.press/v97/houlsby19a.html); pretrained-feature distortion during fine-tuning is studied by [Kumar et al.](https://arxiv.org/abs/2202.10054). Neither these precedents nor this experiment establishes a novel counting contribution.
