# ALIGNN backbone transfer report

Phase A verdict: **BACKBONE_TRANSFER_GAIN**

Phase-A Graph/Early/Hard Macro NMAE: 1.000000 / 0.935459 / 0.953088.
Phase-A Hard improved 5/5 tasks; hard bound pass=True.

## Phase A
| task                   |   graph_valid_mae |   early_valid_mae |   hard_valid_mae |   graph_test_mae |   early_test_mae |   hard_test_mae |   early_gain_vs_graph_pct |   hard_gain_vs_graph_pct |   hard_gain_vs_early_pct |   early_nmae |   hard_nmae |
|:-----------------------|------------------:|------------------:|-----------------:|-----------------:|-----------------:|----------------:|--------------------------:|-------------------------:|-------------------------:|-------------:|------------:|
| Jarvis-Bandgap_MBJ     |         0.265593  |         0.268106  |        0.259242  |        0.268131  |        0.246552  |       0.252113  |                   8.04797 |                 5.97404  |                 -2.25545 |     0.91952  |    0.94026  |
| Jarvis-FormationEnergy |         0.0367422 |         0.0358401 |        0.0359535 |        0.0373422 |        0.0363964 |       0.0368013 |                   2.53274 |                 1.44849  |                 -1.11242 |     0.974673 |    0.985515 |
| Jarvis-BulkModulusKv   |         9.28398   |         9.07591   |        9.55098   |        9.39096   |        9.18595   |       9.37696   |                   2.18302 |                 0.149082 |                 -2.07933 |     0.97817  |    0.998509 |
| MP-Bandgap             |         0.233953  |         0.207708  |        0.210757  |        0.224292  |        0.19493   |       0.205278  |                  13.0909  |                 8.47738  |                 -5.30849 |     0.869091 |    0.915226 |
| MP-FormationEnergy     |         0.025783  |         0.0247946 |        0.024896  |        0.0316873 |        0.0296544 |       0.0293402 |                   6.41571 |                 7.40713  |                  1.05939 |     0.935843 |    0.925929 |

## Phase B
10-task Macro NMAE=0.964608; gain=3.54%; improved=10/10.
Electronic/Energy/Mechanical NMAE=0.939004/0.963046/0.984983.

| task                   | category   |   seed |   graph_valid_mae |   hard_valid_mae |   graph_test_mae |   hard_test_mae |   hard_nmae |   gain_pct | improved   |
|:-----------------------|:-----------|-------:|------------------:|-----------------:|-----------------:|----------------:|------------:|-----------:|:-----------|
| Jarvis-Bandgap_MBJ     | Electronic |     42 |         0.265593  |        0.259242  |        0.268131  |       0.252113  |    0.94026  |   5.97404  | True       |
| Jarvis-Bandgap_OPT     | Electronic |     42 |         0.127343  |        0.120961  |        0.13098   |       0.125941  |    0.961525 |   3.84746  | True       |
| Jarvis-FormationEnergy | Energy     |     42 |         0.0367422 |        0.0359535 |        0.0373422 |       0.0368013 |    0.985515 |   1.44849  | True       |
| Jarvis-TotalEnergy     | Energy     |     42 |         0.0385704 |        0.0381522 |        0.0404068 |       0.0395055 |    0.977693 |   2.23065  | True       |
| Jarvis-BulkModulusKv   | Mechanical |     42 |         9.28398   |        9.55098   |        9.39096   |       9.37696   |    0.998509 |   0.149082 | True       |
| Jarvis-ShearModulusGv  | Mechanical |     42 |         8.7498    |        8.77716   |        9.20867   |       9.11146   |    0.989444 |   1.05561  | True       |
| MP-Bandgap             | Electronic |     42 |         0.233953  |        0.210757  |        0.224292  |       0.205278  |    0.915226 |   8.47738  | True       |
| MP-FormationEnergy     | Energy     |     42 |         0.025783  |        0.024896  |        0.0316873 |       0.0293402 |    0.925929 |   7.40713  | True       |
| MP-BulkModuli          | Mechanical |     42 |         0.0376175 |        0.0400034 |        0.0384105 |       0.0375453 |    0.977474 |   2.25256  | True       |
| MP-ShearModuli         | Mechanical |     42 |         0.0678911 |        0.068306  |        0.0716871 |       0.0698594 |    0.974504 |   2.54962  | True       |

## Phase C
3-seed Macro NMAE=0.970285 ± 0.006585; gain=2.97%.
Tasks improved in at least 2/3 seeds: 9/10.

| task | graph mean MAE | graph sample SD | GPSR mean MAE | GPSR sample SD | mean GN-MAE | mean gain (%) | improved seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jarvis-Bandgap_MBJ | 0.26328509 | 0.0063493106 | 0.25372409 | 0.0030830891 | 0.96399203 | 3.6008 | 3/3 |
| Jarvis-Bandgap_OPT | 0.13080003 | 0.0015607094 | 0.12388205 | 0.0019478758 | 0.94723629 | 5.27637 | 3/3 |
| Jarvis-FormationEnergy | 0.036841527 | 0.00053433096 | 0.03615622 | 0.00055898479 | 0.98141498 | 1.8585 | 3/3 |
| Jarvis-TotalEnergy | 0.040390504 | 0.00065636082 | 0.039626583 | 0.00051725602 | 0.98112445 | 1.88756 | 3/3 |
| Jarvis-BulkModulusKv | 9.3133509 | 0.13946263 | 9.3587894 | 0.035693377 | 1.0049918 | -0.499185 | 2/3 |
| Jarvis-ShearModulusGv | 9.082342 | 0.1145857 | 9.1232228 | 0.01036573 | 1.0046157 | -0.461567 | 1/3 |
| MP-Bandgap | 0.22501728 | 0.0015441391 | 0.20347643 | 0.0017779303 | 0.90429552 | 9.57045 | 3/3 |
| MP-FormationEnergy | 0.030988033 | 0.0012328891 | 0.029483569 | 0.00046593934 | 0.95218956 | 4.78104 | 3/3 |
| MP-BulkModuli | 0.038719499 | 0.0010238698 | 0.038092847 | 0.00090958349 | 0.98455454 | 1.54455 | 2/3 |
| MP-ShearModuli | 0.072008682 | 0.00033562407 | 0.070457445 | 0.00085546153 | 0.97843691 | 2.15631 | 3/3 |

**Final verdict: BACKBONE_AGNOSTIC_GRAPH_PRESERVING_GAIN**
