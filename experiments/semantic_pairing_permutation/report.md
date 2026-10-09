# Semantic Pairing Permutation Control

## Design

This exploratory post-hoc experiment uses frozen, validation-selected bounded-GPSR checkpoints. For each test set it compares (i) the correct crystal–MatBERT pairing, (ii) the same checkpoint with the semantic residual disabled, and (iii) 20 fixed-point-free permutations of the test embeddings. The permutations preserve the empirical embedding distribution while destroying sample correspondence. No training, checkpoint selection, task selection, or hyperparameter adjustment uses these results.

## Results by run

| task                   |   seed |   correct_mae |   off_mae |   shuffle_mean_mae |   pairing_advantage_pct_vs_shuffle |   correct_gain_pct_vs_off |   fraction_samples_pairing_helped |   permutation_p_one_sided |
|:-----------------------|-------:|--------------:|----------:|-------------------:|-----------------------------------:|--------------------------:|----------------------------------:|--------------------------:|
| Jarvis-Bandgap_MBJ     |      7 |     0.270518  | 0.291336  |          0.303471  |                            10.8587 |                   7.14578 |                          0.588559 |                  0.047619 |
| Jarvis-BulkModulusKv   |      7 |     9.07043   | 9.88892   |         10.3621    |                            12.4652 |                   8.27685 |                          0.680894 |                  0.047619 |
| Jarvis-FormationEnergy |      7 |     0.0316923 | 0.0392246 |          0.0396502 |                            20.0702 |                  19.203   |                          0.711645 |                  0.047619 |
| MP-Bandgap             |      7 |     0.191952  | 0.236585  |          0.255576  |                            24.8947 |                  18.8657  |                          0.637597 |                  0.047619 |
| Jarvis-Bandgap_MBJ     |     42 |     0.264172  | 0.287451  |          0.308595  |                            14.3951 |                   8.0982  |                          0.621012 |                  0.047619 |
| Jarvis-Bandgap_OPT     |     42 |     0.12161   | 0.139908  |          0.155828  |                            21.9587 |                  13.0781  |                          0.604522 |                  0.047619 |
| Jarvis-BulkModulusKv   |     42 |     8.75642   | 9.57315   |         10.1407    |                            13.6504 |                   8.53143 |                          0.692581 |                  0.047619 |
| Jarvis-FormationEnergy |     42 |     0.0314828 | 0.0385284 |          0.0408565 |                            22.943  |                  18.2868  |                          0.733357 |                  0.047619 |
| MP-Bandgap             |     42 |     0.199384  | 0.236546  |          0.24858   |                            19.7907 |                  15.7102  |                          0.622093 |                  0.047619 |
| MP-BulkModuli          |     42 |     0.0364735 | 0.0476815 |          0.0573989 |                            36.4561 |                  23.506   |                          0.778731 |                  0.047619 |
| Jarvis-Bandgap_MBJ     |   1234 |     0.267449  | 0.287705  |          0.311665  |                            14.187  |                   7.04054 |                          0.606711 |                  0.047619 |
| Jarvis-BulkModulusKv   |   1234 |     8.98653   | 9.594     |         10.1909    |                            11.8183 |                   6.33175 |                          0.69563  |                  0.047619 |
| Jarvis-FormationEnergy |   1234 |     0.0317067 | 0.0387388 |          0.0408316 |                            22.3477 |                  18.1526  |                          0.727436 |                  0.047619 |
| MP-Bandgap             |   1234 |     0.210516  | 0.243268  |          0.262793  |                            19.8928 |                  13.4634  |                          0.651163 |                  0.047619 |

## Multi-seed task summary

| task                   |   runs |   mean_pairing_advantage_pct |   sample_sd_pairing_advantage_pct |   min_pairing_advantage_pct |   max_pairing_advantage_pct |   mean_correct_gain_pct_vs_off |   mean_fraction_samples_pairing_helped |
|:-----------------------|-------:|-----------------------------:|----------------------------------:|----------------------------:|----------------------------:|-------------------------------:|---------------------------------------:|
| Jarvis-Bandgap_MBJ     |      3 |                      13.1469 |                          1.98442  |                     10.8587 |                     14.3951 |                        7.42817 |                               0.605427 |
| Jarvis-Bandgap_OPT     |      1 |                      21.9587 |                                   |                     21.9587 |                     21.9587 |                       13.0781  |                               0.604522 |
| Jarvis-BulkModulusKv   |      3 |                      12.6446 |                          0.929127 |                     11.8183 |                     13.6504 |                        7.71334 |                               0.689702 |
| Jarvis-FormationEnergy |      3 |                      21.787  |                          1.51624  |                     20.0702 |                     22.943  |                       18.5475  |                               0.724146 |
| MP-Bandgap             |      3 |                      21.5261 |                          2.91773  |                     19.7907 |                     24.8947 |                       16.0131  |                               0.636951 |
| MP-BulkModuli          |      1 |                      36.4561 |                                   |                     36.4561 |                     36.4561 |                       23.506   |                               0.778731 |

Correct pairing had lower MAE than every one of the 20 derangements in all 14/14 task–seed runs. Accordingly, each run attains the smallest possible one-sided Monte Carlo p-value with 20 permutations, 1/21 = 0.0476. These per-run values are descriptive and are not presented as multiplicity-corrected confirmatory tests.

Across the six seed-42 tasks, the mean MAE advantage of correct pairing relative to shuffled pairing was 21.532% and was positive on 6/6 tasks. Across the four three-seed tasks, the task-mean advantage ranged from 12.645% to 21.787%. Correct pairing also outperformed semantic OFF in 14/14 runs, whereas shuffled pairing was worse than semantic OFF in 14/14 runs.

The sample-level effect is heterogeneous: the fraction of samples benefiting from correct rather than mean-shuffled pairing ranges from 0.589 to 0.779. Thus the aggregate signal is consistent, but it is not a claim that every crystal benefits.

MP-Bulk at seed 42 shows the largest pairing advantage (36.456%). This does not contradict the independently trained CartNet comparison: the present result says that the trained GPSR checkpoint uses sample-matched semantic information more effectively than mismatched or disabled semantics; it does not say that GPSR must outperform an independently optimized graph-only backbone.

## Interpretation

The results support semantic specificity within the trained GPSR models: preserving crystal–text correspondence matters beyond merely supplying embeddings with the same marginal distribution. They also show that mismatched semantic interventions can be actively harmful compared with disabling the intervention. Because this is a test-label, within-checkpoint, post-hoc analysis, it is an exploratory association and cannot establish a causal advantage over independently trained graph-only models or justify model selection.

## Integrity checks

All 14 required task–seed runs, 280 derangements, sample tables, and prediction arrays are present. All permutations have zero fixed points. Recomputed formal predictions agree with archived predictions within 1.02e-06 in the model's standardized-output scale, below the 1e-5 tolerance. Raw physical-unit discrepancies are retained for transparency.
