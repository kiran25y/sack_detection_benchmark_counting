# V14 completed: FSC regression and LVIS improvement

Verified October 23, 2026 KST. `full-multidomain-v14-recovery-v1` exited 0, OOMKilled=false. All 56 rounds and 25,592 optimizer steps completed. All eight result-file hashes, 97 immutable release files, training and LVIS manifest hashes, native prediction labels/groups/coverage and finite count metrics verified. Presentations match the frozen plan: FSC 96,005, LVIS 106,026 and historical sacks 2,537. No training or evaluation was repeated by this verification.

| Metric | Retained V11 | V14 |
| --- | ---: | ---: |
| FSC validation MAE, 1,286 queries | 12.493685 | 15.720033 |
| FSC RMSE | 52.766016 | 72.968441 |
| FSC bias | -1.163091 | -12.950387 |
| FSC GAME1 / GAME2 | 14.188315 / 17.085177 | 17.116818 / 19.439423 |
| FSC class-macro MAE | 11.902788 | 14.243371 |
| FSC sparse MAE, 603 queries | 2.560506 | 1.812888 |
| FSC dense MAE, 115 queries | 86.175434 | 122.708749 |
| LVIS validation MAE, separate 1,181 queries | 39.295701 | 14.577309 |
| LVIS RMSE | 74.880873 | 24.520342 |
| LVIS bias | 29.206115 | -6.805834 |
| LVIS dense MAE, 38 queries | 74.402080 | 75.876869 |

FSC V14-minus-V11 MAE is +3.226348, paired image interval [1.430269, 5.345897] and category-component interval [0.026666, 7.155214]. The useful-gain gate fails. Only sparse non-regression passes; RMSE, GAME1 and class-macro regress. V14 dense bias is -116.412931 and these 115 rows account for 69.8038% of its total absolute error. Below-9 and below-8 outcomes are both false.

FSC gaps to CountGD++ global and GeCo2 global are +6.883330 and +6.303237. Their respective image intervals are [4.714259, 8.997216] and [3.892908, 9.453215]; category intervals [2.860918, 12.425653] and [2.320579, 11.511203]. Both global comparators remain better under this reused native protocol.

LVIS improves by -24.718392 MAE versus frozen V11, with paired image interval [-28.169558, -21.517156]. This is useful evidence for the declared multidomain variant, not independent-domain confirmation: LVIS is now a training domain. Its dense subset does not improve, and its opposing bias changes must not be described as accuracy by themselves. Never pool FSC and LVIS MAEs. Foreground sack MAE is still unavailable because exact labels and recording separation remain unresolved.

Decision: retain V11 for the FSC/general-counter development baseline, preserve V14's LVIS benefit and FSC failure, and do not launch a mixture sweep, seed repeat or multiplier. The fixed optimizer budget does not equal identical source exposure or total computation. This data comparison does not isolate a new mechanism.

Verification independently recomputes MAE/RMSE/bias and count strata from saved predictions. GAME, class-macro and grouped intervals are read from the hash-verified completed reports, not independently recomputed in this check. DGX timestamps have the documented clock offset and are not interpreted as client UTC. Next mechanism investigation uses prior fitting arrays and remains separate from validation-based model selection.
