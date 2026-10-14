# Targeted CartNet Multi-seed Robustness Report

Original ten-task seed-42 conclusion remains unchanged: 6/10 improved, Macro GN-MAE 1.0050, no aggregate gain.

| task                 |   cartnet_mean_mae |   cartnet_sample_std |   gpsr_mean_mae |   gpsr_sample_std |   mean_paired_gain_pct |   paired_gain_sample_std |   improved_runs |   num_seeds | verdict             |
|:---------------------|-------------------:|---------------------:|----------------:|------------------:|-----------------------:|-------------------------:|----------------:|------------:|:--------------------|
| Jarvis-Bandgap_MBJ   |          0.248032  |           0.00161889 |       0.240181  |       0.0121384   |                3.14672 |                  5.39514 |               2 |           3 | STABLE_POSITIVE     |
| Jarvis-BulkModulusKv |          8.58727   |           0.225999   |       8.72494   |       0.262462    |               -1.69889 |                  5.64796 |               2 |           3 | NOT_STABLE_POSITIVE |
| MP-BulkModuli        |          0.0314261 |           0.00110509 |       0.0343259 |       0.000347194 |               -9.29268 |                  2.79988 |               0 |           3 | STABLE_NEGATIVE     |

This is an exploratory targeted three-task subset; no ten-task three-seed macro is computed.
