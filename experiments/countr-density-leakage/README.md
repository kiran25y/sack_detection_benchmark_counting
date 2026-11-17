# CounTR density-loss test: verified result — November 5, 2026

**Both C and primary L completed300/300updates; relative and absolute gates FAILED for L. Full training is not justified.**

Verified at 2026-11-05T16:28:45.614412+09:00. Both containers exited0 with no OOM or guard stop; the controller completed and exited. Execution finished approximately14:03KST. All88release files, calibration/initial identities,600sequential optimizer calls,1,200identical inputs per arm and final checkpoint hashes/literal metadata were verified. A separate read-only calculation checked260B/P/C/Ldensity arrays for MAE/RMSE/bias/GAME, agreeing within7.11e-15;20fine spatial checks used an independent pairwise-distance calculation. Model/Adam tensor finiteness is a worker check, not independent tensor deserialization.

B=frozen pretrained; P=completed original-MSE pretrained decoder; C=new total-count penalty; L=new density-leakage penalty. B/P outputs were reused.

| Development metric | B | P | C | L primary |
|---|---:|---:|---:|---:|
| MAE | 9.0393 | 7.6703 | 7.9650 | 15.3912 |
| RMSE | 12.5801 | 10.2154 | 10.8655 | 19.3714 |
| Bias | 0.9357 | 3.2381 | 2.5306 | -15.3898 |
| GAME1 | 11.8858 | 10.1703 | 9.6467 | 15.3932 |
| GAME2 | 14.2161 | 13.7056 | 13.4277 | 15.3978 |
| GAME3 | 15.9639 | 15.9580 | 15.1649 | 15.4017 |
| Spatial mass precision@8 (%) | 12.1055 | 6.8145 | 7.2005 | 9.6869 |
| Spatial mass recall@8 (%) | 13.7566 | 7.4809 | 7.9634 | 0.0280 |
| Low-count MAE | 2.4885 | 12.8044 | 8.6684 | 3.6617 |
| Higher-count MAE | 11.9402 | 7.2589 | 9.1979 | 22.8664 |
| Mean empty prediction | 4.2671 | 5.8242 | 3.8445 | 0.0033 |

## Why the gate failed

L predicted only **1.0353 total sacks across23development images containing355 reviewed sacks**. Every development prediction was below0.219. Fitting also collapsed:1.4052predicted mass versus338labels, fittingMAE8.0149. Thus this is failure to retain positive density, not only poor transfer to development.

L MAE rises by+7.7209 versusP,+7.4262 versusC,and+6.3519 versusB. Higher-count(12–36)MAE rises to22.8664, versusP7.2589 andC9.1979. Its spatial mass recall is0.0280%, versusP7.4809%,C7.9634%,B13.7566%. The apparent spatial precision of9.6869% is the fraction of a tiny predicted mass; it cannot compensate for almost no recalled sack mass. These spatial mass metrics are not point-detection F1.

Empty-image mean0.003327 passes the empty protection and absolute<=0.5check, but it is obtained alongside almost total suppression on positive images. Low-countMAE3.6617 improves versusP12.8044/C8.6684 yet worsens versusfrozenB2.4885 by+1.1733; therefore low-count protection fails. Low-countGAME1/2/3 pass protection, but that does not rescue missed sacks.

All three required overall-MAE gains fail; the higher-count gain/protections fail. AggregateRMSE/GAME1/GAME2regress against every comparator. GAME3regresses againstC, despite a small reduction againstB/P. Both recording-groupMAEs fail protection against all comparators. Spatial recall fails against all; precision fails againstB. Relative gateFAIL. Absolute gateFAIL: MAE15.3912>7.7174,fittingMAE8.0149>2,precision0.0969<0.5,recall0.000280<0.5. Only the absolute empty check passes.

C is a useful control, not a replacement winner: MAE7.9650 is+0.2947 worse thanP7.6703; higher-countMAE is+1.9390 worse. C improves aggregateGAME1/2/3 and reduces low/empty overcount versusP, but low-countMAE8.6684 remains worse thanB and empty mean3.8445 remains above0.5.

## Recording groups

| Group | B MAE | P MAE | C MAE | L MAE |
|---|---:|---:|---:|---:|
| March12,15images | 8.2411 | 8.0068 | 8.0962 | 17.8663 |
| February27,8images | 10.5360 | 7.0394 | 7.7192 | 10.7505 |

## Interpretation and decision

The leakage objective in this frozen recipe suppressed true and false density together instead of concentrating useful mass at reviewed points. C and L had similar initial parameter-gradient norms(~14), but trajectories and clipping differ:68.7%ofCupdates and24.0%ofLupdates clipped. The fixed output-gradient normalization never guaranteed equal parameter updates. These observations identify the failed behavior, not a uniquely proven optimizer cause or a theorem against other density methods.

Preserve the original reporting rule max(raw,0)/60. L signed raw total is−58.2222 and negative mass59.2575; these remain diagnostics and cannot be substituted for the frozen metrics. No threshold, coefficient, alternate arm, extra seed or longer fit is selected after scoring.

**Stop this density-leakage repair recipe under the authorized scope.** No full training or new experiment was launched. The result uses42fitting frames/3groups and23reuseddevelopment frames/2groups,with3low-count,15higher-count and5emptydevelopment images. Five reserved recording groups remain untouched. It is not independent test accuracy, FSCdense/extreme evidence, a novel contribution or paper readiness.

Independent analysis · Supplemental verification · Result receipt

## Published files

The `results` directory contains the saved analysis, protocol or verification files for this experiment.

The `code` directory contains the corresponding worker or evaluation scripts. Training code uses the original model dependencies and dataset paths.

The [scientific output bundle](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/sack-experiment-outputs.zip) contains 148 files under the folder countr-density-leakage/. These include the run’s locally collected numerical outputs. See the file list below.

[Archived experiment code](frozen-source.zip) preserves the source tree from the worker package. Machine-specific paths and third-party model dependencies in these historical workers belong to their original training environment.
