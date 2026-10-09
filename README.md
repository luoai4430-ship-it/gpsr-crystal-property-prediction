# GPSR: bounded semantic residual learning for crystal property prediction

This repository contains the code and compact result artifacts for
**Bounded Semantic Residual Learning for Graph-Preserving Crystal Property Prediction**.

Graph-Preserving Semantic Residual learning (GPSR) injects a frozen 768-dimensional
MatBERT crystal-description representation after the first valid graph block. A
node gate and channel-wise residual scale control the update, and the paper's
bounded variant applies a per-crystal Frobenius norm cap with `tau = 0.10`.

The repository was curated from the final server workspace. Raw datasets,
processed graph caches, language-model weights, training checkpoints, and
third-party source trees are intentionally excluded.

## Main audited results

| Backbone / study | Evidence | Macro GN-MAE | Matched improvements |
| --- | --- | ---: | ---: |
| SFTGNN + GPSR | 10 tasks, seeds 42/7/1234 | `0.9664 +/- 0.0090` | `28/30` |
| ALIGNN + GPSR | 10 tasks, seeds 42/7/1234 | `0.9703 +/- 0.0066` | `26/30` |
| CartNet + GPSR | complete seed-42 screen | `1.0050` | `6/10` |

The standard deviations above are sample standard deviations. Negative results
are retained. The canonical SFTGNN paired rows are in
`experiments/final_10task/summary/multiseed_paired_results.csv`.

## Repository layout

```text
data/                    dataset loaders and graph construction
model/                   SFTGNN and GPSR model definitions
src/                     shared utilities
scripts/                 data preparation and release verification
tests/                   unit and release-integrity tests
experiments/final_10task/
                         SFTGNN runners and audited summaries
experiments/backbone_transfer_alignn/
                         ALIGNN adapter, runners, and summaries
experiments/cartnet_gpsr/
                         CartNet adapter, post-hoc analysis, and summaries
experiments/final_confirmatory/
                         SMM/unbounded/bounded and targeted CartNet checks
docs/                    method, reproduction, and result notes
metadata/                release manifest and upstream provenance
```

## Environment

The recorded environment uses Python 3.10, PyTorch 2.2.2, CUDA 12.1,
PyTorch Geometric 2.5.2, and `torch-scatter` 2.1.2.

```bash
conda env create -f environment.yml
conda activate gpsr-crystal
python -m pytest -q
```

Adapt the CUDA packages to the installed driver when necessary.

## Data preparation

Each task snapshot uses the following layout:

```text
snapshot/
  cif/
    id_prop.csv
    <sample_id>.cif
  MatBert_embeddings.pt
  description.csv
```

`id_prop.csv` contains `sample_id,target` rows without a header. Text embeddings
must be keyed by exactly the same sample IDs as the graph samples. The repository
does not redistribute JARVIS or Materials Project data, Robocrystallographer
outputs, or MatBERT weights.

Description and embedding helpers:

```bash
python scripts/generate_descriptions.py /path/to/cif \
  --output derived/description.csv --no-embed

python scripts/embed_descriptions.py derived/description.csv \
  --output derived/MatBert_embeddings.pt \
  --model-name /path/to/matbert-base-cased
```

## One GPSR run

The following command reproduces the SFTGNN bounded GPSR configuration for one
task after the dataset and frozen MatBERT embedding cache are prepared:

```bash
python run.py \
  --task train_and_test \
  --target Jarvis-Bandgap_MBJ \
  --model gpsrnodegate-early-bounded \
  --use_cif_csv \
  --cif_dir /path/to/snapshot/cif \
  --text_emb_path /path/to/snapshot/MatBert_embeddings.pt \
  --num_epoch 300 \
  --batch_size 32 \
  --seed 42 \
  --lr 1e-3 \
  --max_lr 1e-3 \
  --alpha_init 1e-3 \
  --max_residual_ratio 0.10 \
  --inject_after 1
```

Checkpoint selection is based on minimum validation MAE. Formal test evaluation
must be performed only once for the validation-selected checkpoint.

## Strong-backbone adapters

- ALIGNN scripts require the upstream ALIGNN implementation and the paths
  described in `experiments/backbone_transfer_alignn/preflight.json`.
- CartNet scripts require official CartNet commit
  `1c794049db48badfe71f7030b58ea50665ba9fd5`.
- Third-party source code is not vendored in this repository. See
  `docs/UPSTREAM_DEPENDENCIES.md`.

## Result integrity

Run the compact consistency audit before publishing derived tables:

```bash
python scripts/audit_release_results.py
```

It checks the 30 canonical SFTGNN task-seed rows, the `28/30` improvement count,
the sample-standard-deviation macro summary, ALIGNN transfer results, and the
14-run frozen-checkpoint semantic-pairing control.

The semantic-pairing experiment and compact sample-level outputs are under
`experiments/semantic_pairing_permutation/`. Large prediction arrays and runtime
logs remain in the archived experiment workspace rather than this release.

## License

Project code is released under the Apache License 2.0. Dataset records,
pretrained models, and upstream backbones retain their original terms.
