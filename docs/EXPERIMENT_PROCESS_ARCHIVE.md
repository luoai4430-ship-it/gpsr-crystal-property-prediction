# Paper experiment process archive

This directory preserves the server-side process records used for the experiments
reported in *Bounded Semantic Residual Learning for Graph-Preserving Crystal
Property Prediction*. The files were copied without changing their contents and
are verified against `metadata/server_experiment_process_manifest.csv`.

## Paper-to-directory map

| Paper evidence | Server experiment | Archive directory |
| --- | --- | --- |
| SFTGNN ten-task benchmark, seeds 42/7/1234 | `final_10task` | `experiments/process_archive/final_10task/` |
| SMM, unbounded GPSR, bounded GPSR, and targeted confirmation | `final_confirmatory` | `experiments/process_archive/final_confirmatory/` |
| ALIGNN transfer and semantic comparator | `backbone_transfer_alignn` | `experiments/process_archive/backbone_transfer_alignn/` |
| Official CartNet reproduction, seed-42 screen, multiseed confirmation, and direction analysis | `cartnet_gpsr` | `experiments/process_archive/cartnet_gpsr/` |

The archive contains launch and analysis scripts, frozen configurations, preflight
records, queue state, exit markers, standard-output and run logs, raw JSON/CSV/JSONL
metrics, formal-test predictions, split manifests, diagnostic tables, plots, reports,
and experiment-local code snapshots. It contains 1,132 files totaling 257,397,570
bytes before Git compression.

## Excluded artifacts

Normal Git hosting is unsuitable for model weights and large reproducible caches.
The transfer therefore excluded checkpoint and tensor binaries (`.pt`, `.pth`,
`.ckpt`, `.bin`, and `.safetensors`), processed/cache trees, compiled Python
bytecode, vendored third-party source trees, and any remaining file above the
95 MiB release threshold. No retained file exceeds the GitHub 100 MiB object
limit.

Every excluded server path, byte size, and exclusion reason is recorded in
`metadata/server_experiment_excluded_artifacts.csv`. The excluded artifacts total
185 files and 2,537,285,151 bytes. The audit-facing metrics, predictions, logs,
split records, and best-checkpoint selection evidence remain in this repository;
the omitted weights and data caches remain part of the original server workspace.

## Integrity

`metadata/server_experiment_process_manifest.csv` records the original relative
path, byte size, and SHA-256 digest of every retained file. Run:

```bash
python scripts/verify_release.py
```

The verifier maps each source path into `experiments/process_archive/`, checks all
1,132 file sizes and hashes, audits the canonical published summaries, and then
checks the repository-wide release checksum manifest.
