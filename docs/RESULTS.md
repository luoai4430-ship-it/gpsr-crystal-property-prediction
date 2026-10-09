# Audited result index

## SFTGNN

- Canonical paired rows: `experiments/final_10task/summary/multiseed_paired_results.csv`
- Task summary: `experiments/final_10task/summary/multiseed_task_summary.csv`
- Macro-by-seed: `experiments/final_10task/summary/multiseed_macro_by_seed.csv`
- Result: Macro GN-MAE `0.9664 +/- 0.0090`, sample SD, `28/30` matched improvements.

The two regressions are MP Formation Energy at seed 7 and JARVIS MBJ band gap
at seed 1234.

## ALIGNN

- Paired rows: `experiments/backbone_transfer_alignn/summary/alignn_multiseed_paired.csv`
- Report: `experiments/backbone_transfer_alignn/summary/alignn_transfer_report.md`
- Result: Macro GN-MAE `0.9703 +/- 0.0066`, sample SD, `26/30` matched improvements.

## CartNet

- Complete screen: `experiments/cartnet_gpsr/summary/cartnet_seed1_10task.csv`
- Report: `experiments/cartnet_gpsr/summary/cartnet_gpsr_report.md`
- Result: Macro GN-MAE `1.0050`, `6/10` improvements in the complete seed-42 screen.
- Targeted follow-up: `experiments/final_confirmatory/cartnet_targeted_multiseed/`.

## Semantic intervention comparison

`experiments/final_confirmatory/semantic_intervention/macro_summary.csv` records
the matched six-task seed-42 comparison of graph-only, SMM, unbounded residual,
and bounded GPSR. These results support an accuracy-control trade-off rather
than a universal-improvement claim.
