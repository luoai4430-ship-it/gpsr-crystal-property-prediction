# Archived output correction analysis

This is an exploratory post-hoc analysis of the 14 frozen **SFTGNN bounded-GPSR** checkpoint evaluations stored in the parent experiment. It executes no model inference or training. It analyses prediction-level corrections, not hidden-space residual direction. Formal independently trained graph-only comparisons retain their original interpretation.

## Definitions and exact decomposition

For each crystal, let `e = y_true - y_OFF`, `d = y_ON - y_OFF`, and `g = abs(e) - abs(e-d)`. Positive `g` means that the semantic intervention reduces absolute error in that same co-adapted checkpoint.

- Target-directed correction: `e*d > 0`.
- Helpful target-directed correction: `e*d > 0` and `abs(d) < 2*abs(e)`.
- Harmful target-directed overshoot: `e*d > 0` and `abs(d) > 2*abs(e)`.
- Target-directed tie: `e*d > 0` and `abs(d) == 2*abs(e)`.
- Reverse-direction correction: `e*d < 0`; every such correction increases absolute error.
- OFF-exact shift: `e == 0` and `abs(d) > 0`; it also increases absolute error.
- No shift: `d == 0`.

The last six categories partition every stored correction. Category contributions are `mean(g * category_indicator)` over **all N crystals**, so their sum is the total OFF-minus-ON MAE change. They are not within-category mean gains. Fractions and equality tests use the exact archived float64 values. `run_summary.csv` records all category fractions and contributions for correct and shuffled pairing, including zero/tie cases.

The correction amplitude is `mean(abs(d))`; its normalized version divides by OFF MAE within the same run. Shuffled amplitude, direction, benefit, and contributions are first averaged across the 20 interventions for each crystal and then across crystals. This avoids treating 20 correlated interventions on one crystal as 20 independent observations.

## Results

| task                   |   seed |   correct_abs_shift_over_off_mae |   shuffled_to_correct_abs_shift_ratio |   correct_direction_correct |   shuffled_direction_correct |   correct_benefit_vs_off |   correct_aligned_overshoot_fraction |   correct_reverse_fraction |
|:-----------------------|-------:|---------------------------------:|--------------------------------------:|----------------------------:|-----------------------------:|-------------------------:|-------------------------------------:|---------------------------:|
| Jarvis-Bandgap_MBJ     |      7 |                          0.29113 |                               1.05683 |                     0.58361 |                      0.53468 |                  0.52530 |                              0.05831 |                    0.41639 |
| Jarvis-BulkModulusKv   |      7 |                          0.28590 |                               1.00871 |                     0.65955 |                      0.50447 |                  0.59248 |                              0.06707 |                    0.34045 |
| Jarvis-FormationEnergy |      7 |                          0.47798 |                               0.90153 |                     0.72385 |                      0.59028 |                  0.64113 |                              0.08272 |                    0.27615 |
| MP-Bandgap             |      7 |                          0.43683 |                               1.08556 |                     0.60853 |                      0.54612 |                  0.53682 |                              0.07171 |                    0.39147 |
| Jarvis-Bandgap_MBJ     |     42 |                          0.28188 |                               1.13609 |                     0.61551 |                      0.54246 |                  0.55666 |                              0.05886 |                    0.38449 |
| Jarvis-Bandgap_OPT     |     42 |                          0.39049 |                               1.08503 |                     0.61457 |                      0.56832 |                  0.53939 |                              0.07518 |                    0.38543 |
| Jarvis-BulkModulusKv   |     42 |                          0.28647 |                               0.99815 |                     0.65803 |                      0.49827 |                  0.59248 |                              0.06555 |                    0.34197 |
| Jarvis-FormationEnergy |     42 |                          0.48500 |                               0.86207 |                     0.71703 |                      0.54346 |                  0.64023 |                              0.07680 |                    0.28297 |
| MP-Bandgap             |     42 |                          0.44652 |                               1.01477 |                     0.61047 |                      0.54002 |                  0.53876 |                              0.07171 |                    0.38953 |
| MP-BulkModuli          |     42 |                          0.54108 |                               1.13576 |                     0.72556 |                      0.49717 |                  0.63293 |                              0.09262 |                    0.27444 |
| Jarvis-Bandgap_MBJ     |   1234 |                          0.29875 |                               1.16879 |                     0.61771 |                      0.53339 |                  0.54510 |                              0.07261 |                    0.38229 |
| Jarvis-BulkModulusKv   |   1234 |                          0.27553 |                               1.01504 |                     0.66717 |                      0.50869 |                  0.59400 |                              0.07317 |                    0.33283 |
| Jarvis-FormationEnergy |   1234 |                          0.47517 |                               0.87609 |                     0.71398 |                      0.55514 |                  0.64095 |                              0.07303 |                    0.28602 |
| MP-Bandgap             |   1234 |                          0.40792 |                               1.03815 |                     0.65116 |                      0.53789 |                  0.58140 |                              0.06977 |                    0.34884 |

Correct pairing moves the output toward the target in 58.36%–72.56% of crystals. It reduces error relative to OFF in 52.53%–64.11%. Correct target direction does not guarantee lower error: 5.83%–9.26% of crystals are shifted in the correct direction but overshoot far enough to worsen absolute error. Reverse-direction corrections account for 27.44%–41.64%.

Direction alignment is higher for correct than shuffled pairing in 14/14 runs. For JARVIS Bulk, the shuffled mean absolute correction is 0.9982–1.0150 times the correct-pairing correction, whereas direction alignment falls from 65.80%–66.72% to 49.83%–50.87%. Thus these checkpoints exhibit an output-correction alignment advantage despite similar **mean absolute output shift**. Equal means do not establish matched per-crystal or hidden-residual magnitudes, so this remains descriptive rather than a magnitude-controlled causal direction experiment.

## Contributions to MAE improvement

Values below are in each target's stored output units. Do not average physical-unit gains across different targets.

| task                   |   seed |   correct_mae_gain_vs_off |   correct_aligned_helpful_mae_gain_contribution |   correct_aligned_overshoot_mae_gain_contribution |   correct_reverse_mae_gain_contribution |   shuffled_mae_gain_vs_off |
|:-----------------------|-------:|--------------------------:|------------------------------------------------:|--------------------------------------------------:|----------------------------------------:|---------------------------:|
| Jarvis-Bandgap_MBJ     |      7 |                   0.02082 |                                         0.04862 |                                          -0.00281 |                                -0.02499 |                   -0.01213 |
| Jarvis-BulkModulusKv   |      7 |                   0.81849 |                                         1.68612 |                                          -0.09504 |                                -0.77259 |                   -0.47317 |
| Jarvis-FormationEnergy |      7 |                   0.00753 |                                         0.01182 |                                          -0.00075 |                                -0.00354 |                   -0.00043 |
| MP-Bandgap             |      7 |                   0.04463 |                                         0.06814 |                                          -0.00471 |                                -0.01880 |                   -0.01899 |
| Jarvis-Bandgap_MBJ     |     42 |                   0.02328 |                                         0.04901 |                                          -0.00261 |                                -0.02312 |                   -0.02114 |
| Jarvis-Bandgap_OPT     |     42 |                   0.01830 |                                         0.03385 |                                          -0.00264 |                                -0.01291 |                   -0.01592 |
| Jarvis-BulkModulusKv   |     42 |                   0.81673 |                                         1.64353 |                                          -0.09519 |                                -0.73161 |                   -0.56751 |
| Jarvis-FormationEnergy |     42 |                   0.00705 |                                         0.01160 |                                          -0.00074 |                                -0.00381 |                   -0.00233 |
| MP-Bandgap             |     42 |                   0.03716 |                                         0.06579 |                                          -0.00417 |                                -0.02446 |                   -0.01203 |
| MP-BulkModuli          |     42 |                   0.01121 |                                         0.01659 |                                          -0.00166 |                                -0.00372 |                   -0.00972 |
| Jarvis-Bandgap_MBJ     |   1234 |                   0.02026 |                                         0.04879 |                                          -0.00542 |                                -0.02312 |                   -0.02396 |
| Jarvis-BulkModulusKv   |   1234 |                   0.60747 |                                         1.47136 |                                          -0.10338 |                                -0.76051 |                   -0.59693 |
| Jarvis-FormationEnergy |   1234 |                   0.00703 |                                         0.01152 |                                          -0.00080 |                                -0.00369 |                   -0.00209 |
| MP-Bandgap             |   1234 |                   0.03275 |                                         0.06142 |                                          -0.00300 |                                -0.02567 |                   -0.01952 |

Helpful aligned shifts contribute positively, while reverse shifts and overshoots subtract from that gain. The full run table retains tie and OFF-exact contributions, which makes the decomposition exact. The shuffled net contribution is negative in every run even though some shuffled interventions happen to improve individual crystals.

Reverse-direction corrections account for 69.14%–89.89% of the total harmful absolute-error contribution under correct pairing. This refers to harmful error magnitude, not the fraction of harmed samples. It identifies reverse output shifts as the larger observed loss component; it does not establish a training remedy or hidden-space cause.

## Two distinct benefit fractions

The earlier report's 58.86%–77.87% range compares **correct absolute error against each crystal's mean absolute error over 20 shuffled predictions**. The 52.53%–64.11% range here compares **correct absolute error against OFF absolute error**. Both are correct for their respective comparators. Mean shuffled absolute error is not the absolute error of a mean shuffled prediction; this analysis uses the former. Neither number is an improvement fraction against an independently trained graph baseline.

## Paired descriptive bootstrap intervals

The table gives pairing MAE advantage as `100*(mean shuffled error - mean correct error)/mean shuffled error`, plus direction-fraction difference (a fraction, not percentage points). There are 2000 percentile bootstrap replicates per run. Each replicate samples N crystals with replacement using the same indices for correct and all shuffled predictions. All 20 stored permutations are retained; permutation×crystal pairs are never pooled as independent samples. These intervals condition on the realized 20 permutations and omit Monte Carlo uncertainty across new permutations. They describe sampling variation in the stored test set, not independent confirmatory significance tests. Shared compositions, related structures, fixed splits, and repeated training seeds can introduce dependencies the crystal bootstrap does not model. No multiplicity correction or cross-run pooled p-value is claimed.

| task                   |   seed |   pairing_advantage_pct_vs_shuffled |   pairing_advantage_pct_ci95_low |   pairing_advantage_pct_ci95_high |   direction_correct_difference |   direction_correct_difference_ci95_low |   direction_correct_difference_ci95_high |
|:-----------------------|-------:|------------------------------------:|---------------------------------:|----------------------------------:|-------------------------------:|----------------------------------------:|-----------------------------------------:|
| Jarvis-Bandgap_MBJ     |      7 |                            10.85868 |                          8.05758 |                          13.78822 |                        0.04893 |                                 0.02670 |                                  0.07167 |
| Jarvis-BulkModulusKv   |      7 |                            12.46522 |                         10.06096 |                          14.92293 |                        0.15508 |                                 0.13382 |                                  0.17630 |
| Jarvis-FormationEnergy |      7 |                            20.07023 |                         18.65668 |                          21.53810 |                        0.13356 |                                 0.12217 |                                  0.14447 |
| MP-Bandgap             |      7 |                            24.89465 |                         17.94544 |                          31.73580 |                        0.06240 |                                 0.02393 |                                  0.10059 |
| Jarvis-Bandgap_MBJ     |     42 |                            14.39509 |                         11.42891 |                          17.28396 |                        0.07305 |                                 0.04992 |                                  0.09604 |
| Jarvis-Bandgap_OPT     |     42 |                            21.95869 |                         19.03080 |                          24.85537 |                        0.04625 |                                 0.03446 |                                  0.05788 |
| Jarvis-BulkModulusKv   |     42 |                            13.65040 |                         11.22174 |                          16.35559 |                        0.15976 |                                 0.13828 |                                  0.18154 |
| Jarvis-FormationEnergy |     42 |                            22.94297 |                         21.38366 |                          24.35434 |                        0.17357 |                                 0.16134 |                                  0.18484 |
| MP-Bandgap             |     42 |                            19.79069 |                         12.99617 |                          27.04943 |                        0.07045 |                                 0.03207 |                                  0.11134 |
| MP-BulkModuli          |     42 |                            36.45606 |                         31.72083 |                          40.94441 |                        0.22839 |                                 0.19074 |                                  0.26861 |
| Jarvis-Bandgap_MBJ     |   1234 |                            14.18704 |                         11.12855 |                          17.54167 |                        0.08432 |                                 0.06138 |                                  0.10756 |
| Jarvis-BulkModulusKv   |   1234 |                            11.81832 |                          9.74903 |                          14.05138 |                        0.15849 |                                 0.13554 |                                  0.18087 |
| Jarvis-FormationEnergy |   1234 |                            22.34769 |                         20.87535 |                          23.81169 |                        0.15884 |                                 0.14698 |                                  0.17012 |
| MP-Bandgap             |   1234 |                            19.89285 |                         14.35919 |                          26.10788 |                        0.11328 |                                 0.07683 |                                  0.15243 |

Pairing MAE advantage intervals lie above zero in 14/14 runs, and direction-difference intervals lie above zero in 14/14 runs. These are descriptive intervals rather than a new confirmatory verdict.

## Provenance and limits

`source_checksums.csv` records SHA256 and size for each of the 42 input NPZ/summary/sample files. `run_summary.csv` repeats each predictions NPZ checksum. The script verifies finite arrays, 20×N shuffled shape, unique sample IDs, exact category partitions, contribution sums, and consistency with the archived summary and sample CSV. Run `python analyze.py` to regenerate all outputs deterministically.

The observable `d` is the final **output difference** after the downstream graph blocks and head. It is not the hidden injected tensor and does not identify its angle, gate, or norm. Neither output alignment nor ON-versus-OFF benefit proves that correct semantics improve an independently trained graph-only model, explains the training trajectory, or licenses test-informed deployment selection. The informative next model-level control is a label-free conditional or residual-norm-matched semantic mismatch, with donor indices and actual embedding-change statistics saved.
