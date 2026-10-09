# Reproducibility notes

## Fixed evaluation rules

1. Use the predefined task split and preserve sample IDs.
2. Select the checkpoint only by validation MAE.
3. Evaluate the formal test split once for the selected checkpoint.
4. Keep graph-only and GPSR runs matched by task, split, and training seed.
5. Compute reported variability across independent training seeds with the
   sample standard deviation (`ddof=1`).

## SFTGNN study

- Tasks: six JARVIS and four Materials Project regression tasks.
- Training seeds: 42, 7, and 1234.
- Frozen text: 768-dimensional MatBERT representation.
- GPSR: early NodeGate, channel-wise alpha initialized at `1e-3`.
- Bound: hard sample-wise Frobenius ratio `tau=0.10`.

The complete paired result table is
`experiments/final_10task/summary/multiseed_paired_results.csv`.

## ALIGNN study

The adapter and runners live in `experiments/backbone_transfer_alignn/`.
The repository stores compact summaries but does not vendor ALIGNN or its
datasets.

## CartNet study

Use official CartNet commit `1c794049db48badfe71f7030b58ea50665ba9fd5`,
cutoff 5.0, all radius neighbors (`max_neighbours=-1`), hidden size 256,
four layers, batch size 64, 500 epochs, Adam at `1e-3`, and OneCycleLR with
`pct_start=0.01`. GPSR is injected after `layers[0]`.

## Excluded artifacts

Raw datasets, processed graph caches, frozen-language-model weights,
checkpoints, and detailed training logs are excluded because they are large,
redistributability-constrained, or regenerable. Aggregate metrics and the code
needed to regenerate them are retained.
