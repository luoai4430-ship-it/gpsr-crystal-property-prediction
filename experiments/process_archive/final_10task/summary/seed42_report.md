# Final 10-task benchmark: Seed 42

Final Hard-0.10 improves 10/10 tasks versus Graph, with Macro NMAE 0.956853.
Stage-C trigger: PASS (requires >=7/10 and Macro NMAE < 0.98).

## Macro

| model           |   macro_normalized_mae |   avg_gain_pct |   improved_vs_graph |
|:----------------|-----------------------:|---------------:|--------------------:|
| Graph           |               1        |        0       |                   0 |
| TextResidual    |               0.973491 |        2.65088 |                   8 |
| Early NodeGate  |               0.960537 |        3.94625 |                   9 |
| Final Hard-0.10 |               0.956853 |        4.31465 |                  10 |

## Category macro

| model           |   Electronic |   Energy |   Mechanical |   Overall |    Old_6 |    New_4 |
|:----------------|-------------:|---------:|-------------:|----------:|---------:|---------:|
| Graph           |     1        | 1        |     1        |  1        | 1        | 1        |
| TextResidual    |     0.95672  | 1.00082  |     0.965571 |  0.973491 | 0.972816 | 0.974503 |
| Early NodeGate  |     0.943056 | 0.99003  |     0.951529 |  0.960537 | 0.957354 | 0.965312 |
| Final Hard-0.10 |     0.935076 | 0.971141 |     0.962471 |  0.956853 | 0.951635 | 0.96468  |

## Correct vs shuffled on repaired MP text

| task               |   correct_text_mae |   shuffled_text_mae |   correct_gain_pct | correct_better   |
|:-------------------|-------------------:|--------------------:|-------------------:|:-----------------|
| MP-Bandgap         |          0.192624  |            0.209563 |            8.08332 | True             |
| MP-FormationEnergy |          0.0255261 |            0.027619 |            7.578   | True             |
