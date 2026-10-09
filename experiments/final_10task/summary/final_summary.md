# Final 10-task experiment report

**Verdict: STRONG_FINAL_MODEL**

Text audit: 10/10 PASS.
Seed 42: Final Hard-0.10 improves 10/10 tasks; Macro NMAE = 0.956853.
Multi-seed: completed (3 seeds); Macro NMAE = 0.966372.
Seeds 2025 and 2026 were canceled by user request; robustness scope is seeds 42, 7, and 1234.

## Seed-42 category macro

| model           |   Electronic |   Energy |   Mechanical |   Overall |    Old_6 |    New_4 |
|:----------------|-------------:|---------:|-------------:|----------:|---------:|---------:|
| Graph           |     1        | 1        |     1        |  1        | 1        | 1        |
| TextResidual    |     0.95672  | 1.00082  |     0.965571 |  0.973491 | 0.972816 | 0.974503 |
| Early NodeGate  |     0.943056 | 0.99003  |     0.951529 |  0.960537 | 0.957354 | 0.965312 |
| Final Hard-0.10 |     0.935076 | 0.971141 |     0.962471 |  0.956853 | 0.951635 | 0.96468  |

## Robustness by task

| task                   |   graph_mean_mae |   graph_std_mae |   final_mean_mae |   final_std_mae |   mean_paired_ratio |   std_paired_ratio |   relative_gain_pct |   improved_seeds_count |   num_seeds |
|:-----------------------|-----------------:|----------------:|-----------------:|----------------:|--------------------:|-------------------:|--------------------:|-----------------------:|------------:|
| Jarvis-Bandgap_MBJ     |        0.277082  |     0.00458577  |        0.267956  |     0.0045391   |            0.967346 |         0.0237933  |            3.26536  |                      2 |           3 |
| Jarvis-Bandgap_OPT     |        0.131574  |     0.00272437  |        0.12884   |     0.00333343  |            0.979206 |         0.0141124  |            2.07939  |                      3 |           3 |
| Jarvis-FormationEnergy |        0.0325717 |     0.000278337 |        0.0317481 |     0.00033367  |            0.974855 |         0.0175564  |            2.51455  |                      3 |           3 |
| Jarvis-TotalEnergy     |        0.034863  |     0.000150951 |        0.0342518 |     0.000229984 |            0.982492 |         0.0086083  |            1.75081  |                      3 |           3 |
| Jarvis-BulkModulusKv   |        9.24824   |     0.0933997   |        9.13941   |     0.0571166   |            0.98828  |         0.00574813 |            1.17201  |                      3 |           3 |
| Jarvis-ShearModulusGv  |        8.67583   |     0.0644685   |        8.54503   |     0.0635451   |            0.984959 |         0.00829323 |            1.50412  |                      3 |           3 |
| MP-Bandgap             |        0.215767  |     0.00237655  |        0.19535   |     0.00384459  |            0.90534  |         0.0124214  |            9.46599  |                      3 |           3 |
| MP-FormationEnergy     |        0.0261416 |     0.000426804 |        0.0262772 |     0.000590115 |            1.00535  |         0.0242129  |           -0.535295 |                      2 |           3 |
| MP-BulkModuli          |        0.0387675 |     0.00035161  |        0.0361056 |     0.000213692 |            0.931383 |         0.00660002 |            6.86172  |                      3 |           3 |
| MP-ShearModuli         |        0.0731811 |     0.00103983  |        0.0690955 |     0.000955795 |            0.94451  |         0.0251547  |            5.54899  |                      3 |           3 |

## Repaired MP text correspondence

| task               |   correct_text_mae |   shuffled_text_mae |   correct_gain_pct | correct_better   |
|:-------------------|-------------------:|--------------------:|-------------------:|:-----------------|
| MP-Bandgap         |          0.192624  |            0.209563 |            8.08332 | True             |
| MP-FormationEnergy |          0.0255261 |            0.027619 |            7.578   | True             |

## Hard residual control

All ten tasks satisfy max rho_final <= 0.1001.
