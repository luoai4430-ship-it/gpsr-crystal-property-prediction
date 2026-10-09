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

## Semantic-pairing permutation control

`experiments/semantic_pairing_permutation/task_summary.csv` contains a
post-hoc, frozen-checkpoint control that compares correctly paired MatBERT
embeddings with semantic OFF and 20 within-test-set derangements. Correct
pairing had lower MAE than every derangement in all 14 task-seed runs and was
better than semantic OFF in all 14 runs. The six-task seed-42 mean advantage
relative to shuffled pairing was 21.5323%; the task-level three-seed mean
advantages ranged from 12.6446% to 21.7869% on the four repeated tasks.

This result establishes semantic-pairing specificity within the trained GPSR
checkpoints. It is exploratory and does not replace independently trained
graph-only comparisons. In particular, the large within-checkpoint pairing
advantage for MP-Bulk does not imply that GPSR beats an independently optimized
CartNet model on that task.

## Prediction-level correction decomposition

`experiments/semantic_pairing_permutation/output_correction_analysis/report.md`
analyses the archived SFTGNN predictions without model inference or training.
For a fixed checkpoint, it decomposes the output correction `y_ON-y_OFF` into
helpful target-directed shifts, harmful overshoots, reverse shifts, and ties.
Correct pairing moves predictions toward the target on 58.36%–72.56% of samples,
and reduces absolute error relative to OFF on 52.53%–64.11%. Reverse shifts
account for 69.14%–89.89% of the total harmful error contribution.

For JARVIS Bulk across three training seeds, shuffled mean absolute output
corrections are 0.9982–1.0150 times those under correct pairing. Direction
alignment nevertheless falls from 65.80%–66.72% to 49.83%–50.87%. This is a
descriptive output-level finding: mean amplitudes do not match each crystal's
hidden residual norm. Paired crystal bootstrap intervals are descriptive,
conditional on the stored 20 permutations and fixed test split. Repeated seeds
do not supply independent test datasets.

The compact release includes the analysis script, run/task summaries, aggregate,
report, and raw-source checksums. Recomputing this analysis requires restoring
the archived `runs/seed_*/<task>/predictions.npz`, `summary.json`, and
`sample_metrics.csv` under the parent experiment; these arrays are not distributed
in this compact repository.

## Composition-conditioned semantic pairing

`experiments/composition_pairing_control/report.md` records 14 frozen SFTGNN
checkpoints, with 20 within-group donor assignments under each of two fixed
restrictions. Correct pairing has a positive MAE advantage in 12/14 runs under
exact reduced-composition matching and 13/14 under element-set matching.

On exact-composition subsets, JARVIS formation-energy advantages are
7.07%–7.27% across three training seeds; JARVIS bulk advantages are
6.07%–10.24%. Formation-energy descriptive crystal-bootstrap intervals are
above zero in all three seeds; bulk intervals cross zero in two seeds.
J-MBJ seed 1234 is slightly adverse (-0.16%), so the conditional benefit is
not universal. MP band gap has only 12 eligible crystals and one distinct
same-composition donor map, making that primary control weak. MP bulk's
seed-42 advantage is 29.01% on 45 eligible crystals, with limited coverage.

Exact-composition coverage ranges from 2.33% to 12.18%; element-set coverage
from 17.44% to 30.74%. These exploratory subset controls support
correspondence-sensitive information beyond stoichiometry on some tasks.
They do not quantify a causal percentage split of composition and structural
information, establish population-level significance, or replace independent
graph-only comparisons. The global contextual donors came from the whole test
set, while conditional donors came from eligible groups. Crystal-bootstrap
intervals do not account for composition-cluster or donor dependence.

The compact release includes code, fixed protocol, coverage, all compact run
summaries, validation attestations, and the figure. Full donor matrices,
sample-group maps, sample-level metrics, predictions, and failure evidence
remain in the archived experiment workspace. Run `finalize.py --root` against
that full archive to repeat the raw-array audit.
