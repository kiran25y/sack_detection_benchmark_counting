# Residual decomposition result — October 28, 2026

**Interpretation:** the uniform component accounts for most of the local downward count response, but removing it does not provide the required useful gain. Dense MAE changes +0.0120%, extreme MAE −0.0236%, and sparse RMSE +0.3311% relative to the starting checkpoint. Dense spatial protections also fail. Full training has not started. See the measured decision and limitations.

**Verified FAILED gate at 2026-10-28 16:52:40 KST.** 288/550 correlated scientific checklist checks failed. These counts are not independent hypothesis tests.

The corrected run completed two disposable scientific Adam updates and four native guard updates, with exactly173 science and64 guard batch forwards. All19 endpoints cover the same65 fitting queries, including all7 extreme examples. Thirteen CPU tests passed. Both native containers and the controller exited0 without OOM. The archive independently verifies127 release files,321 selected feature files, source/class/checkpoint and historical identities. No trained model was persisted and no full fit started.

V1 remains preserved after a cross-device norm-control assertion with zero Adam calls. V2 calculates control norms separately on CPU/CUDA; it changes no scientific primary, loss, labels, readouts, optimizer budget, metric threshold or native tolerance.

## Full endpoint tables

### all

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 79.330583 | 195.228375 | -43.972553 | 81.658435 | 90.558330 | 102.139697 | 1.153975 | 55.620283 |
| summary_full | 79.408611 | 195.742913 | -44.629370 | 81.810837 | 90.680219 | 102.159126 | 1.153816 | 55.582188 |
| ordered_full | 79.412747 | 195.742810 | -44.624964 | 81.813962 | 90.684859 | 102.157012 | 1.153808 | 55.581927 |
| summary_mean | 79.413801 | 195.780625 | -44.645598 | 81.826176 | 90.691096 | 102.173632 | 1.153832 | 55.583822 |
| summary_centered | 79.323662 | 195.198137 | -43.954448 | 81.655226 | 90.559131 | 102.136966 | 1.153919 | 55.625472 |
| summary_negative_full | 79.227669 | 194.707550 | -43.342479 | 81.502280 | 90.470164 | 102.114222 | 1.154069 | 55.631401 |
| summary_negative_mean | 79.228238 | 194.679339 | -43.324162 | 81.502067 | 90.462810 | 102.114482 | 1.154078 | 55.634682 |
| summary_negative_centered | 79.333512 | 195.251184 | -43.985712 | 81.665230 | 90.570139 | 102.139680 | 1.153973 | 55.612125 |
| summary_matched_full | 79.331874 | 195.272828 | -44.036082 | 81.664389 | 90.557938 | 102.130738 | 1.153922 | 55.610197 |
| summary_matched_mean | 79.332701 | 195.264513 | -44.030728 | 81.669018 | 90.561421 | 102.135075 | 1.153905 | 55.614029 |
| summary_matched_centered | 79.323625 | 195.202324 | -43.955924 | 81.655970 | 90.558538 | 102.135787 | 1.153908 | 55.622753 |
| ordered_mean | 79.408723 | 195.780299 | -44.657577 | 81.819287 | 90.691740 | 102.170228 | 1.153788 | 55.577683 |
| ordered_centered | 79.325728 | 195.204828 | -43.954878 | 81.651062 | 90.552207 | 102.131154 | 1.153952 | 55.617471 |
| ordered_negative_full | 79.233270 | 194.726726 | -43.339365 | 81.505426 | 90.470177 | 102.118483 | 1.154084 | 55.638029 |
| ordered_negative_mean | 79.225565 | 194.681948 | -43.320156 | 81.492239 | 90.460341 | 102.107764 | 1.154072 | 55.643331 |
| ordered_negative_centered | 79.326898 | 195.237769 | -43.987901 | 81.663483 | 90.557981 | 102.133982 | 1.153973 | 55.612881 |
| ordered_matched_full | 79.325588 | 195.260632 | -44.029489 | 81.662263 | 90.554057 | 102.130888 | 1.153913 | 55.607204 |
| ordered_matched_mean | 79.331573 | 195.268006 | -44.032278 | 81.668626 | 90.565176 | 102.138580 | 1.153899 | 55.612754 |
| ordered_matched_centered | 79.322078 | 195.210231 | -43.961918 | 81.649901 | 90.554426 | 102.128978 | 1.153921 | 55.616672 |

### sparse_0_24

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.919527 | 3.117169 | -0.272975 | 3.504993 | 4.639897 | 6.575392 | 1.448242 | 1.975363 |
| summary_full | 1.931909 | 3.187169 | -0.328510 | 3.493088 | 4.629864 | 6.554293 | 1.447342 | 1.991195 |
| ordered_full | 1.930422 | 3.182989 | -0.328773 | 3.496530 | 4.630869 | 6.552689 | 1.447341 | 1.991659 |
| summary_mean | 1.927682 | 3.176127 | -0.318481 | 3.498741 | 4.634878 | 6.561720 | 1.447455 | 1.987719 |
| summary_centered | 1.917106 | 3.124036 | -0.285801 | 3.500052 | 4.636810 | 6.571802 | 1.447911 | 1.976519 |
| summary_negative_full | 1.913110 | 3.056190 | -0.230354 | 3.507355 | 4.648887 | 6.594886 | 1.448619 | 1.966741 |
| summary_negative_mean | 1.912170 | 3.061607 | -0.235338 | 3.503148 | 4.643996 | 6.586911 | 1.448566 | 1.964836 |
| summary_negative_centered | 1.919114 | 3.111708 | -0.272544 | 3.503801 | 4.637044 | 6.574890 | 1.448251 | 1.976919 |
| summary_matched_full | 1.920469 | 3.126338 | -0.283857 | 3.503324 | 4.639502 | 6.572495 | 1.447974 | 1.979642 |
| summary_matched_mean | 1.920813 | 3.125732 | -0.282164 | 3.504096 | 4.640393 | 6.574601 | 1.447916 | 1.980000 |
| summary_matched_centered | 1.918836 | 3.125044 | -0.283502 | 3.501980 | 4.639343 | 6.572888 | 1.447880 | 1.977846 |
| ordered_mean | 1.933262 | 3.178693 | -0.315374 | 3.497974 | 4.638610 | 6.566469 | 1.447350 | 1.992853 |
| ordered_centered | 1.918974 | 3.127491 | -0.286120 | 3.503327 | 4.633633 | 6.569574 | 1.447969 | 1.978531 |
| ordered_negative_full | 1.917266 | 3.063168 | -0.226547 | 3.510639 | 4.651627 | 6.599627 | 1.448726 | 1.968763 |
| ordered_negative_mean | 1.918518 | 3.067227 | -0.236591 | 3.507387 | 4.647832 | 6.586730 | 1.448564 | 1.972783 |
| ordered_negative_centered | 1.919098 | 3.110674 | -0.269498 | 3.505808 | 4.636862 | 6.577922 | 1.448316 | 1.974281 |
| ordered_matched_full | 1.918227 | 3.123630 | -0.283166 | 3.502993 | 4.636299 | 6.570327 | 1.447932 | 1.977017 |
| ordered_matched_mean | 1.921709 | 3.126742 | -0.282227 | 3.503976 | 4.639254 | 6.572783 | 1.447899 | 1.980527 |
| ordered_matched_centered | 1.921811 | 3.125660 | -0.281435 | 3.503745 | 4.637194 | 6.572123 | 1.447955 | 1.979336 |

### medium_25_74

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 8.619029 | 15.219950 | -2.635805 | 12.317182 | 14.913470 | 20.422186 | 1.373673 | 9.106602 |
| summary_full | 8.682705 | 15.502042 | -2.875720 | 12.316383 | 14.899126 | 20.369200 | 1.374581 | 9.181013 |
| ordered_full | 8.691579 | 15.511088 | -2.873856 | 12.323900 | 14.908392 | 20.373229 | 1.374526 | 9.189848 |
| summary_mean | 8.671295 | 15.479048 | -2.857114 | 12.318915 | 14.892547 | 20.369921 | 1.374481 | 9.168092 |
| summary_centered | 8.620917 | 15.251110 | -2.653990 | 12.313449 | 14.917856 | 20.422194 | 1.373864 | 9.110042 |
| summary_negative_full | 8.583317 | 14.967327 | -2.401496 | 12.371048 | 14.946234 | 20.481534 | 1.373328 | 9.058398 |
| summary_negative_mean | 8.587309 | 14.987912 | -2.412599 | 12.360000 | 14.945125 | 20.483346 | 1.373416 | 9.064660 |
| summary_negative_centered | 8.609187 | 15.197834 | -2.620371 | 12.326139 | 14.918175 | 20.426078 | 1.373643 | 9.099398 |
| summary_matched_full | 8.617524 | 15.259189 | -2.665248 | 12.314090 | 14.905066 | 20.411450 | 1.373791 | 9.105408 |
| summary_matched_mean | 8.617297 | 15.257908 | -2.669005 | 12.316144 | 14.904827 | 20.412751 | 1.373780 | 9.105940 |
| summary_matched_centered | 8.617578 | 15.247405 | -2.656090 | 12.314947 | 14.915664 | 20.422138 | 1.373880 | 9.106237 |
| ordered_mean | 8.677395 | 15.469152 | -2.846788 | 12.329879 | 14.896560 | 20.370672 | 1.374474 | 9.172174 |
| ordered_centered | 8.627015 | 15.258114 | -2.653412 | 12.315890 | 14.920660 | 20.419946 | 1.373860 | 9.115053 |
| ordered_negative_full | 8.574584 | 14.958637 | -2.400493 | 12.364014 | 14.942013 | 20.478027 | 1.373272 | 9.047497 |
| ordered_negative_mean | 8.581722 | 14.987596 | -2.421464 | 12.354860 | 14.949322 | 20.483064 | 1.373366 | 9.059818 |
| ordered_negative_centered | 8.603944 | 15.189539 | -2.620225 | 12.329198 | 14.912139 | 20.413667 | 1.373556 | 9.091448 |
| ordered_matched_full | 8.616367 | 15.264114 | -2.675770 | 12.313389 | 14.906659 | 20.407262 | 1.373818 | 9.106527 |
| ordered_matched_mean | 8.615280 | 15.258688 | -2.673356 | 12.314515 | 14.902731 | 20.409854 | 1.373778 | 9.104310 |
| ordered_matched_centered | 8.620102 | 15.251783 | -2.654085 | 12.315047 | 14.918981 | 20.421757 | 1.373785 | 9.109420 |

### medium_75_149

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 14.238094 | 16.661762 | 14.238094 | 14.888323 | 18.826899 | 26.224969 | 0.889601 | 11.542811 |
| summary_full | 14.126773 | 16.543538 | 14.126773 | 14.766111 | 18.743394 | 26.164571 | 0.889199 | 11.402510 |
| ordered_full | 14.150094 | 16.569945 | 14.150094 | 14.785704 | 18.740188 | 26.169792 | 0.889168 | 11.429699 |
| summary_mean | 14.121189 | 16.545999 | 14.121189 | 14.756741 | 18.726009 | 26.152045 | 0.889220 | 11.401278 |
| summary_centered | 14.257106 | 16.685001 | 14.257106 | 14.906267 | 18.832589 | 26.220985 | 0.889680 | 11.560446 |
| summary_negative_full | 14.380046 | 16.798357 | 14.380046 | 15.019691 | 18.933252 | 26.296105 | 0.889891 | 11.713782 |
| summary_negative_mean | 14.397104 | 16.803067 | 14.397104 | 15.036200 | 18.942532 | 26.299223 | 0.889969 | 11.742193 |
| summary_negative_centered | 14.251584 | 16.680647 | 14.251584 | 14.888297 | 18.828921 | 26.225898 | 0.889563 | 11.549669 |
| summary_matched_full | 14.241520 | 16.665295 | 14.241520 | 14.886952 | 18.816904 | 26.205861 | 0.889589 | 11.555264 |
| summary_matched_mean | 14.238640 | 16.663363 | 14.238640 | 14.882138 | 18.817733 | 26.204361 | 0.889557 | 11.551933 |
| summary_matched_centered | 14.250749 | 16.677314 | 14.250749 | 14.899678 | 18.827001 | 26.209883 | 0.889654 | 11.552116 |
| ordered_mean | 14.117695 | 16.539152 | 14.117695 | 14.756864 | 18.716855 | 26.140688 | 0.889179 | 11.391860 |
| ordered_centered | 14.236712 | 16.653107 | 14.236712 | 14.882687 | 18.812648 | 26.205314 | 0.889725 | 11.546721 |
| ordered_negative_full | 14.384001 | 16.805484 | 14.384001 | 15.018802 | 18.925125 | 26.306991 | 0.889877 | 11.710302 |
| ordered_negative_mean | 14.408319 | 16.819143 | 14.408319 | 15.044763 | 18.961830 | 26.310903 | 0.890011 | 11.741427 |
| ordered_negative_centered | 14.253844 | 16.686856 | 14.253844 | 14.898595 | 18.825743 | 26.230765 | 0.889552 | 11.550732 |
| ordered_matched_full | 14.240645 | 16.664760 | 14.240645 | 14.891940 | 18.818077 | 26.212114 | 0.889601 | 11.545162 |
| ordered_matched_mean | 14.240300 | 16.665089 | 14.240300 | 14.883973 | 18.820315 | 26.204804 | 0.889571 | 11.557297 |
| ordered_matched_centered | 14.252099 | 16.674762 | 14.252099 | 14.893899 | 18.815494 | 26.214270 | 0.889648 | 11.557204 |

### dense_150_899

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 100.774459 | 182.150628 | -32.664356 | 100.854967 | 117.862573 | 135.362416 | 0.989372 | 127.707976 |
| summary_full | 100.615442 | 182.310838 | -33.421434 | 100.702699 | 117.614385 | 135.112565 | 0.989158 | 127.683900 |
| ordered_full | 100.624138 | 182.310794 | -33.404700 | 100.710570 | 117.630290 | 135.113747 | 0.989147 | 127.679173 |
| summary_mean | 100.616197 | 182.301302 | -33.435734 | 100.704102 | 117.606189 | 135.106870 | 0.989152 | 127.685112 |
| summary_centered | 100.797850 | 182.166212 | -32.639398 | 100.882883 | 117.907506 | 135.385119 | 0.989382 | 127.732067 |
| summary_negative_full | 100.865304 | 181.949465 | -31.929974 | 100.980331 | 118.198331 | 135.581629 | 0.989627 | 127.652407 |
| summary_negative_mean | 100.881269 | 181.965984 | -31.920692 | 101.006216 | 118.197950 | 135.593738 | 0.989620 | 127.658785 |
| summary_negative_centered | 100.773964 | 182.145460 | -32.671498 | 100.852598 | 117.870051 | 135.348243 | 0.989396 | 127.695202 |
| summary_matched_full | 100.762557 | 182.165509 | -32.749860 | 100.841748 | 117.842701 | 135.337175 | 0.989385 | 127.711934 |
| summary_matched_mean | 100.773673 | 182.166830 | -32.733839 | 100.854017 | 117.849212 | 135.342147 | 0.989399 | 127.712958 |
| summary_matched_centered | 100.797883 | 182.168450 | -32.641626 | 100.882915 | 117.904594 | 135.383512 | 0.989380 | 127.729303 |
| ordered_mean | 100.591667 | 182.295026 | -33.451430 | 100.678869 | 117.600975 | 135.102013 | 0.989124 | 127.668681 |
| ordered_centered | 100.786591 | 182.155893 | -32.634446 | 100.868508 | 117.885272 | 135.366933 | 0.989395 | 127.711578 |
| ordered_negative_full | 100.863793 | 181.949336 | -31.927700 | 100.985703 | 118.200881 | 135.594039 | 0.989635 | 127.643320 |
| ordered_negative_mean | 100.865574 | 181.951271 | -31.918854 | 100.986429 | 118.190572 | 135.585259 | 0.989632 | 127.652104 |
| ordered_negative_centered | 100.762048 | 182.139603 | -32.700231 | 100.843785 | 117.853088 | 135.348161 | 0.989404 | 127.691246 |
| ordered_matched_full | 100.765145 | 182.162811 | -32.733805 | 100.847478 | 117.848037 | 135.345395 | 0.989382 | 127.703826 |
| ordered_matched_mean | 100.770538 | 182.167844 | -32.733301 | 100.852475 | 117.855507 | 135.350189 | 0.989398 | 127.705429 |
| ordered_matched_centered | 100.776865 | 182.154056 | -32.656613 | 100.861343 | 117.883924 | 135.354701 | 0.989388 | 127.705492 |

### dense_150_plus

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 170.568223 | 292.034551 | -100.407826 | 173.282643 | 190.737842 | 211.931062 | 0.929973 | 191.450102 |
| summary_full | 170.735650 | 292.801292 | -101.735941 | 173.658004 | 191.040184 | 212.019929 | 0.930007 | 191.472938 |
| ordered_full | 170.738061 | 292.800694 | -101.731351 | 173.655990 | 191.047358 | 212.013825 | 0.930015 | 191.461037 |
| summary_mean | 170.755287 | 292.858272 | -101.784492 | 173.689552 | 191.066971 | 212.049662 | 0.929995 | 191.498377 |
| summary_centered | 170.549797 | 291.988358 | -100.356064 | 173.276433 | 190.739077 | 211.928238 | 0.929992 | 191.454016 |
| summary_negative_full | 170.324926 | 291.257489 | -99.135149 | 172.885259 | 190.500726 | 211.825336 | 0.929982 | 191.342975 |
| summary_negative_mean | 170.321943 | 291.214759 | -99.090359 | 172.888076 | 190.486080 | 211.830149 | 0.929992 | 191.342526 |
| summary_negative_centered | 170.575676 | 292.068942 | -100.445732 | 173.295613 | 190.764238 | 211.929835 | 0.929980 | 191.431655 |
| summary_matched_full | 170.570279 | 292.100345 | -100.533270 | 173.298489 | 190.742203 | 211.920635 | 0.930001 | 191.444541 |
| summary_matched_mean | 170.572568 | 292.087936 | -100.520545 | 173.308621 | 190.749305 | 211.928764 | 0.930013 | 191.445748 |
| summary_matched_centered | 170.550986 | 291.994782 | -100.358918 | 173.277617 | 190.737913 | 211.927163 | 0.929990 | 191.452608 |
| ordered_mean | 170.738677 | 292.858025 | -101.816322 | 173.670834 | 191.066350 | 212.040849 | 0.929979 | 191.477114 |
| ordered_centered | 170.555256 | 291.998608 | -100.352787 | 173.268879 | 190.728907 | 211.920765 | 0.930019 | 191.441848 |
| ordered_negative_full | 170.336806 | 291.286240 | -99.131961 | 172.892654 | 190.502002 | 211.830573 | 0.929963 | 191.359554 |
| ordered_negative_mean | 170.311181 | 291.218440 | -99.079780 | 172.863124 | 190.472460 | 211.812898 | 0.929988 | 191.344523 |
| ordered_negative_centered | 170.562204 | 292.048925 | -100.453255 | 173.287127 | 190.739852 | 211.918246 | 0.929967 | 191.438731 |
| ordered_matched_full | 170.558316 | 292.082007 | -100.515161 | 173.293163 | 190.734919 | 211.922618 | 0.929996 | 191.434928 |
| ordered_matched_mean | 170.569776 | 292.093128 | -100.522821 | 173.308007 | 190.758696 | 211.938781 | 0.930008 | 191.436630 |
| ordered_matched_centered | 170.544319 | 292.006576 | -100.374749 | 173.263959 | 190.731416 | 211.911651 | 0.930002 | 191.435689 |

### extreme_900_plus

| Endpoint | MAE | RMSE | bias | GAME1 | GAME2 | GAME3 | normalized_blur | class_macro_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 389.920053 | 499.042865 | -313.315874 | 400.912480 | 419.774401 | 452.575378 | 0.743293 | 420.021708 |
| summary_full | 391.113447 | 500.717482 | -316.438677 | 402.946108 | 421.806983 | 453.728786 | 0.744103 | 420.472994 |
| ordered_full | 391.096104 | 500.716085 | -316.472255 | 402.913025 | 421.786715 | 453.699786 | 0.744172 | 420.443840 |
| summary_mean | 391.192426 | 500.866425 | -316.594875 | 403.072394 | 421.943717 | 453.869864 | 0.744073 | 420.581449 |
| summary_centered | 389.770203 | 498.912989 | -313.179871 | 400.799018 | 419.638301 | 452.492327 | 0.743338 | 419.923735 |
| summary_negative_full | 388.626596 | 497.389392 | -310.351414 | 398.872175 | 417.736825 | 451.448414 | 0.742528 | 419.470976 |
| summary_negative_mean | 388.564061 | 497.266731 | -310.195025 | 398.802490 | 417.677344 | 451.430298 | 0.742592 | 419.438472 |
| summary_negative_centered | 389.952488 | 499.132166 | -313.450466 | 400.973657 | 419.860253 | 452.614838 | 0.743243 | 420.007060 |
| summary_matched_full | 389.965977 | 499.185298 | -313.566842 | 401.019673 | 419.854924 | 452.611511 | 0.743366 | 419.998153 |
| summary_matched_mean | 389.940526 | 499.153701 | -313.564479 | 401.023091 | 419.863883 | 452.629562 | 0.743374 | 419.994885 |
| summary_matched_centered | 389.775024 | 498.925997 | -313.184692 | 400.803826 | 419.642630 | 452.492924 | 0.743336 | 419.928280 |
| ordered_mean | 391.200710 | 500.873005 | -316.677412 | 403.074151 | 421.957530 | 453.848620 | 0.744095 | 420.551624 |
| ordered_centered | 389.828203 | 498.949683 | -313.181859 | 400.812901 | 419.666048 | 452.518524 | 0.743411 | 419.959574 |
| ordered_negative_full | 388.680559 | 497.459287 | -310.345354 | 398.885927 | 417.734098 | 451.431109 | 0.742423 | 419.572673 |
| ordered_negative_mean | 388.568804 | 497.292583 | -310.156974 | 398.761309 | 417.644106 | 451.385476 | 0.742536 | 419.463180 |
| ordered_negative_centered | 389.934126 | 499.090358 | -313.391331 | 400.966202 | 419.812537 | 452.567082 | 0.743165 | 420.035079 |
| ordered_matched_full | 389.908282 | 499.143937 | -313.542280 | 400.979601 | 419.807977 | 452.593887 | 0.743353 | 419.972823 |
| ordered_matched_mean | 389.938808 | 499.165125 | -313.575596 | 401.025395 | 419.883007 | 452.645785 | 0.743356 | 419.984367 |
| ordered_matched_centered | 389.813459 | 498.971109 | -313.203177 | 400.815039 | 419.680677 | 452.519209 | 0.743360 | 419.951701 |

## Uniform and spatial contributions

Matching-response means and centered variations are measured before the unchanged nonlinear decoder. Their count effects need not add linearly. Ratios below compare mean-only and full count shifts; they are not percentages of the historical32-step error explained. No source endpoint from that earlier test was retained.

| Input | Group | Full mean count shift | Mean-only shift | Centered-only shift | Mean absolute full-minus-mean | Relative L1 difference | Mean nonlinear interaction |
|---|---|---:|---:|---:|---:|---:|---:|
| summary | all | -0.65681705 | -0.67304537 | 0.01810466 | 0.05047060 | 0.07639484 | -0.00187634 |
| summary | sparse_0_24 | -0.05553524 | -0.04550598 | -0.01282589 | 0.01203618 | 0.20759061 | 0.00279663 |
| summary | medium_25_74 | -0.23991528 | -0.22130890 | -0.01818447 | 0.02368183 | 0.09674599 | -0.00042191 |
| summary | medium_75_149 | -0.11132177 | -0.11690521 | 0.01901118 | 0.03417079 | 0.25009539 | -0.01342773 |
| summary | dense_150_899 | -0.75707730 | -0.77137756 | 0.02495783 | 0.04312411 | 0.05696131 | -0.01065757 |
| summary | dense_150_plus | -1.32811448 | -1.37666584 | 0.05176202 | 0.08958698 | 0.06745426 | -0.00321066 |
| summary | extreme_900_plus | -3.12280273 | -3.27900042 | 0.13600377 | 0.23561314 | 0.07544926 | 0.02019392 |
| ordered | all | -0.65241063 | -0.68502450 | 0.01767495 | 0.06125512 | 0.09319746 | 0.01493891 |
| ordered | sparse_0_24 | -0.05579779 | -0.04239888 | -0.01314538 | 0.01349571 | 0.23056908 | -0.00025353 |
| ordered | medium_25_74 | -0.23805103 | -0.21098280 | -0.01760654 | 0.03974066 | 0.16219775 | -0.00946169 |
| ordered | medium_75_149 | -0.08799998 | -0.12039948 | -0.00138219 | 0.03520711 | 0.29379682 | 0.03378169 |
| ordered | dense_150_899 | -0.74034361 | -0.78707400 | 0.02991035 | 0.06850052 | 0.09252531 | 0.01682004 |
| ordered | dense_150_plus | -1.32352500 | -1.40849620 | 0.05503924 | 0.10700068 | 0.08084523 | 0.02993196 |
| ordered | extreme_900_plus | -3.15638079 | -3.36153739 | 0.13401576 | 0.22800119 | 0.07223501 | 0.07114083 |

Mean residual energy fractions (each active image/level/exemplar observation is dependent): {"summary": {"mean": 0.9876133797291564, "minimum": 0.9401327101895534, "maximum": 0.9976213307795098, "active_level_exemplars": 585}, "ordered": {"mean": 0.9804070924147512, "minimum": 0.8961340661522521, "maximum": 0.9967582958637905, "active_level_exemplars": 585}}.

Maximum absolute centered-channel mean after BF16 casting: {"summary": 1.317384885624051e-06, "ordered": 1.888663973659277e-06}. The residual is centered in FP32; postcast zero mean is not claimed. Matching common norms are computed across six components, both inputs, independently per image/level/exemplar; only shrink factors are used.

## Every extreme example

| Query | Truth | Baseline AE | Ordered full AE | Ordered mean AE | Ordered centered AE | Summary centered AE |
|---|---:|---:|---:|---:|---:|---:|
| 2360.jpg | 1283 | 293.094727 | 295.358704 | 295.626343 | 292.987854 | 292.911682 |
| 7603.jpg | 1400 | 124.849365 | 127.807739 | 127.848877 | 124.598389 | 124.531738 |
| 805.jpg | 1095 | 314.824951 | 320.086487 | 320.378052 | 314.625061 | 314.649658 |
| 2347.jpg | 1912 | 937.386902 | 940.368408 | 940.711060 | 937.016418 | 937.064270 |
| 2775.jpg | 1672 | 791.169800 | 792.867920 | 793.009094 | 791.307495 | 791.167908 |
| 804.jpg | 1001 | 166.563110 | 161.805176 | 161.373291 | 166.920654 | 166.617920 |
| 4128.jpg | 953 | 101.551514 | 99.378296 | 99.458252 | 101.341553 | 101.448242 |

## Complete gate and decision

Failures by reference: {"baseline": 20, "ordered_full": 27, "ordered_mean": 26, "summary_full": 27, "summary_mean": 26, "summary_centered": 28, "ordered_matched_full": 32, "ordered_matched_mean": 26, "summary_matched_full": 24, "summary_matched_mean": 24, "summary_matched_centered": 28}.

Failures by kind: {"gain_1percent": 66, "protection": 195, "extreme_individual_protection": 27}. Every measured component, including signed and magnitude-matched controls, is retained in artifacts/residual_decomposition_v2_20261028/analysis.json and result_outputs.zip. Negative signs are diagnostic controls, never substitute candidates.

This is an activation intervention after one unconstrained adapter update per input. It does not establish how gradients through a centered operator would train over many steps. Small or absent one-step gains do not prove that longer training or fresh initialization cannot work. A full run requires the protected fixed primary to pass, then a separately frozen trainable recipe with fair controls and native recovery checks. The owner has authorized that conditional next step; no additional permission is needed once those requirements are met.

All65 queries are previously seen fitting data, reused in development. No new source validation, class-held, reserved, external or sack accuracy is reported. Retain V11 until an appropriate completed comparison supports replacement.
