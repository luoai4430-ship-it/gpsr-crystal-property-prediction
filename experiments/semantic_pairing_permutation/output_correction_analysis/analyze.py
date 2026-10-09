"""Describe output corrections using archived predictions; no model execution.

Run from any working directory with: python path/to/analyze.py
Only writes files into this script's output_correction_analysis directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import zlib

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261009
GROUPS = ("aligned_helpful", "aligned_overshoot", "aligned_tie", "reverse",
          "off_exact_shift", "no_shift")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correction_metrics(truth: np.ndarray, off: np.ndarray,
                       prediction: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-crystal metrics, averaging permutations within each crystal.

    prediction is [N] for correct pairing or [P,N] for shuffled pairing.
    The definitions are exact in the stored float64 output arrays.
    """
    e = truth - off
    delta = prediction - off
    amplitude = np.abs(delta)
    error = np.abs(truth - prediction)
    gain = np.abs(e) - error
    product = e * delta
    aligned = product > 0
    reverse = product < 0
    twice_error = 2 * np.abs(e)
    masks = {
        "aligned_helpful": aligned & (amplitude < twice_error),
        "aligned_overshoot": aligned & (amplitude > twice_error),
        "aligned_tie": aligned & (amplitude == twice_error),
        "reverse": reverse,
        "off_exact_shift": (e == 0) & (amplitude > 0),
        "no_shift": amplitude == 0,
    }
    partition = sum(mask.astype(int) for mask in masks.values())
    if not np.all(partition == 1):
        raise AssertionError("Correction categories do not partition observations")

    def by_sample(array):
        return array.mean(axis=0) if array.ndim == 2 else array

    metrics = {
        "abs_error": by_sample(error),
        "abs_shift": by_sample(amplitude),
        "direction_correct": by_sample(aligned.astype(float)),
        "benefit_vs_off": by_sample((gain > 0).astype(float)),
        "harm_vs_off": by_sample((gain < 0).astype(float)),
        "tie_vs_off": by_sample((gain == 0).astype(float)),
        "mae_gain_vs_off": by_sample(gain),
        "positive_mae_gain": by_sample(np.maximum(gain, 0)),
        "harmful_mae_loss": by_sample(np.maximum(-gain, 0)),
        "shift_square": by_sample(delta ** 2),
        "error_shift_product": by_sample(e * delta),
    }
    for group, mask in masks.items():
        metrics[group + "_fraction"] = by_sample(mask.astype(float))
        # All contributions use the full test-set denominator N, not group N.
        metrics[group + "_mae_gain_contribution"] = by_sample(gain * mask)
    contribution = sum(metrics[g + "_mae_gain_contribution"] for g in GROUPS)
    if not np.allclose(contribution, metrics["mae_gain_vs_off"], rtol=1e-12, atol=1e-12):
        raise AssertionError("MAE contribution decomposition does not sum")
    return metrics


def paired_bootstrap(correct: dict, shuffled: dict, n: int,
                     seed: int) -> dict[str, float]:
    """Resample crystals jointly; never resample permutation-crystal pairs."""
    differences = {
        "pairing_mae_advantage": shuffled["abs_error"] - correct["abs_error"],
        "direction_correct_difference": correct["direction_correct"] - shuffled["direction_correct"],
        "abs_shift_difference": correct["abs_shift"] - shuffled["abs_shift"],
        "benefit_vs_off_difference": correct["benefit_vs_off"] - shuffled["benefit_vs_off"],
        "overshoot_difference": correct["aligned_overshoot_fraction"] - shuffled["aligned_overshoot_fraction"],
        "reverse_difference": correct["reverse_fraction"] - shuffled["reverse_fraction"],
    }
    names = list(differences)
    # The final column lets each bootstrap ratio retain its own denominator.
    matrix = np.column_stack([differences[name] for name in names] + [shuffled["abs_error"]])
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, BOOTSTRAP_REPLICATES, 64):
        count = min(64, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, n, size=(count, n))
        estimates.append(matrix[indices].mean(axis=1))
    sampled = np.concatenate(estimates)
    out = {}
    for i, name in enumerate(names):
        low, high = np.quantile(sampled[:, i], [.025, .975])
        out[name + "_ci95_low"] = float(low)
        out[name + "_ci95_high"] = float(high)
    low, high = np.quantile(100 * sampled[:, 0] / sampled[:, -1], [.025, .975])
    out["pairing_advantage_pct_ci95_low"] = float(low)
    out["pairing_advantage_pct_ci95_high"] = float(high)
    return out


def main():
    prediction_files = sorted((SOURCE / "runs").glob("seed_*/*/predictions.npz"))
    if len(prediction_files) != 14:
        raise AssertionError(f"Expected 14 archived runs, found {len(prediction_files)}")
    rows, hashes = [], []
    for prediction_path in prediction_files:
        run = prediction_path.parent
        summary_path, sample_path = run / "summary.json", run / "sample_metrics.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sample = pd.read_csv(sample_path)
        arrays = np.load(prediction_path)
        truth, off = arrays["truth"], arrays["off"]
        correct_prediction, shuffled_prediction = arrays["correct"], arrays["shuffled"]
        n = len(truth)
        if shuffled_prediction.shape != (20, n) or correct_prediction.shape != (n,) or off.shape != (n,):
            raise AssertionError(f"Unexpected shape: {prediction_path}")
        if any(not np.all(np.isfinite(arrays[k])) for k in arrays.files):
            raise AssertionError(f"Non-finite archived output: {prediction_path}")
        if not sample.sample_id.is_unique or len(sample) != n:
            raise AssertionError(f"Sample IDs invalid: {sample_path}")
        for name, values in (("y_true", truth), ("y_off", off), ("y_correct", correct_prediction)):
            if not np.allclose(sample[name].to_numpy(), values, rtol=1e-12, atol=1e-12):
                raise AssertionError(f"CSV/NPZ mismatch: {sample_path}, {name}")
        correct = correction_metrics(truth, off, correct_prediction)
        shuffled = correction_metrics(truth, off, shuffled_prediction)
        off_mae = float(np.abs(truth - off).mean())
        row = {"task": summary["task"], "seed": int(summary["seed"]),
               "num_samples": n, "num_permutations": 20, "off_mae": off_mae,
               "source_predictions_sha256": checksum(prediction_path)}
        for prefix, metrics in (("correct", correct), ("shuffled", shuffled)):
            for name, values in metrics.items():
                row[prefix + "_" + name] = float(values.mean())
            row[prefix + "_abs_shift_over_off_mae"] = row[prefix + "_abs_shift"] / off_mae
            row[prefix + "_gain_pct_vs_off"] = 100 * row[prefix + "_mae_gain_vs_off"] / off_mae
            row[prefix + "_reverse_share_of_harmful_mae_loss"] = (
                -row[prefix + "_reverse_mae_gain_contribution"] / row[prefix + "_harmful_mae_loss"])
        row["shuffled_to_correct_abs_shift_ratio"] = row["shuffled_abs_shift"] / row["correct_abs_shift"]
        row["pairing_mae_advantage"] = row["shuffled_abs_error"] - row["correct_abs_error"]
        row["pairing_advantage_pct_vs_shuffled"] = 100 * row["pairing_mae_advantage"] / row["shuffled_abs_error"]
        row["fraction_correct_better_than_mean_shuffled_error"] = float((shuffled["abs_error"] > correct["abs_error"]).mean())
        for name in ("direction_correct", "abs_shift", "benefit_vs_off"):
            row[name + "_difference"] = row["correct_" + name] - row["shuffled_" + name]
        row["overshoot_difference"] = row["correct_aligned_overshoot_fraction"] - row["shuffled_aligned_overshoot_fraction"]
        row["reverse_difference"] = row["correct_reverse_fraction"] - row["shuffled_reverse_fraction"]
        bootstrap_seed = BOOTSTRAP_SEED + int(summary["seed"]) + zlib.adler32(summary["task"].encode())
        row.update(paired_bootstrap(correct, shuffled, n, bootstrap_seed))
        row["bootstrap_seed"] = bootstrap_seed
        for path in (prediction_path, summary_path, sample_path):
            hashes.append({"source_file": path.relative_to(SOURCE).as_posix(),
                           "sha256": checksum(path), "bytes": path.stat().st_size})
        # Check against the previously archived summary, without changing it.
        for key, metric in (("correct_mae", "correct_abs_error"),
                            ("shuffle_mean_mae", "shuffled_abs_error")):
            if not np.isclose(row[metric], summary[key], rtol=1e-12, atol=1e-12):
                raise AssertionError(f"Summary mismatch: {key}")
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["seed", "task"])
    df.to_csv(HERE / "run_summary.csv", index=False)
    pd.DataFrame(hashes).to_csv(HERE / "source_checksums.csv", index=False)
    # Equal weight to training seeds; task means do not pool crystal counts.
    summary_fields = ["off_mae", "correct_abs_error", "shuffled_abs_error",
                      "correct_abs_shift", "shuffled_abs_shift",
                      "correct_abs_shift_over_off_mae", "shuffled_abs_shift_over_off_mae",
                      "correct_mae_gain_vs_off", "shuffled_mae_gain_vs_off",
                      "correct_positive_mae_gain", "shuffled_positive_mae_gain",
                      "correct_harmful_mae_loss", "shuffled_harmful_mae_loss",
                      "correct_reverse_share_of_harmful_mae_loss", "shuffled_reverse_share_of_harmful_mae_loss",
                      "correct_direction_correct", "shuffled_direction_correct",
                      "correct_benefit_vs_off", "shuffled_benefit_vs_off",
                      "correct_aligned_overshoot_fraction", "shuffled_aligned_overshoot_fraction",
                      "correct_reverse_fraction", "shuffled_reverse_fraction",
                      "shuffled_to_correct_abs_shift_ratio", "pairing_advantage_pct_vs_shuffled",
                      "fraction_correct_better_than_mean_shuffled_error"]
    summary_fields.extend(prefix + "_" + group + "_mae_gain_contribution"
                          for prefix in ("correct", "shuffled") for group in GROUPS)
    task_rows = []
    for task, group in df.groupby("task", sort=True):
        row = {"task": task, "num_runs": len(group), "seeds": "/".join(str(x) for x in sorted(group.seed))}
        for name in summary_fields:
            row["mean_" + name] = float(group[name].mean())
            row["sample_sd_" + name] = float(group[name].std(ddof=1)) if len(group) > 1 else None
        task_rows.append(row)
    task_df = pd.DataFrame(task_rows)
    task_df.to_csv(HERE / "task_summary.csv", index=False)
    seed42 = df[df.seed.eq(42)]
    aggregate = {
        "analysis": "Exploratory archived-output correction decomposition",
        "backbone": "SFTGNN", "model_inference": False, "training": False,
        "num_runs": 14, "num_tasks_seed42": 6, "permutations_per_run": 20,
        "analysis_script_sha256": checksum(Path(__file__)),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "base_seed": BOOTSTRAP_SEED,
                      "unit": "test crystal", "type": "paired percentile descriptive interval",
                      "permutations": "all 20 retained and averaged within each crystal",
                      "scope": "conditional on stored checkpoints, test samples, and 20 stored permutations"},
        "correct_vs_off_benefit_fraction_range": [float(df.correct_benefit_vs_off.min()), float(df.correct_benefit_vs_off.max())],
        "correct_vs_mean_shuffled_benefit_fraction_range": [float(df.fraction_correct_better_than_mean_shuffled_error.min()), float(df.fraction_correct_better_than_mean_shuffled_error.max())],
        "correct_direction_fraction_range": [float(df.correct_direction_correct.min()), float(df.correct_direction_correct.max())],
        "correct_aligned_overshoot_fraction_range": [float(df.correct_aligned_overshoot_fraction.min()), float(df.correct_aligned_overshoot_fraction.max())],
        "correct_reverse_fraction_range": [float(df.correct_reverse_fraction.min()), float(df.correct_reverse_fraction.max())],
        "correct_reverse_share_of_harmful_mae_loss_range": [float(df.correct_reverse_share_of_harmful_mae_loss.min()), float(df.correct_reverse_share_of_harmful_mae_loss.max())],
        "runs_direction_fraction_greater_than_shuffled": int((df.direction_correct_difference > 0).sum()),
        "runs_direction_difference_ci95_above_zero": int((df.direction_correct_difference_ci95_low > 0).sum()),
        "runs_pairing_mae_advantage_ci95_above_zero": int((df.pairing_mae_advantage_ci95_low > 0).sum()),
        "seed42_macro_pairing_advantage_pct": float(seed42.pairing_advantage_pct_vs_shuffled.mean()),
        "notes": ["The output correction is y_ON-y_OFF, not the hidden GPSR residual tensor.",
                  "Bootstrap intervals are descriptive; crystals need not be independent if composition/structure groups overlap.",
                  "Training seeds use the fixed split and do not represent independent test sets.",
                  "No test-informed checkpoint selection, tuning, or paper changes."]}
    (HERE / "aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def markdown_view(columns, renamed=None):
        view = df[columns].copy()
        if renamed:
            view = view.rename(columns=renamed)
        return view.to_markdown(index=False, floatfmt=".5f")

    view = markdown_view(["task", "seed", "correct_abs_shift_over_off_mae", "shuffled_to_correct_abs_shift_ratio",
                          "correct_direction_correct", "shuffled_direction_correct", "correct_benefit_vs_off",
                          "correct_aligned_overshoot_fraction", "correct_reverse_fraction"])
    components = markdown_view(["task", "seed", "correct_mae_gain_vs_off", "correct_aligned_helpful_mae_gain_contribution",
                                "correct_aligned_overshoot_mae_gain_contribution", "correct_reverse_mae_gain_contribution",
                                "shuffled_mae_gain_vs_off"])
    intervals = markdown_view(["task", "seed", "pairing_advantage_pct_vs_shuffled", "pairing_advantage_pct_ci95_low",
                               "pairing_advantage_pct_ci95_high", "direction_correct_difference",
                               "direction_correct_difference_ci95_low", "direction_correct_difference_ci95_high"])
    jbulk = df[df.task.eq("Jarvis-BulkModulusKv")]
    report = f"""# Archived output correction analysis

This is an exploratory post-hoc analysis of the 14 frozen **SFTGNN bounded-GPSR** checkpoint evaluations stored in the parent experiment. It executes no model inference or training. It analyses prediction-level corrections, not hidden-space residual direction. Formal independently trained graph-only comparisons retain their original interpretation.

## Definitions and exact decomposition

For each crystal, let `e = y_true - y_OFF`, `d = y_ON - y_OFF`, and `g = abs(e) - abs(e-d)`. Positive `g` means that the semantic intervention reduces absolute error in that same co-adapted checkpoint.

- Target-directed correction: `e*d > 0`.
- Helpful target-directed correction: `e*d > 0` and `abs(d) < 2*abs(e)`.
- Harmful target-directed overshoot: `e*d > 0` and `abs(d) > 2*abs(e)`.
- Target-directed tie: `e*d > 0` and `abs(d) == 2*abs(e)`.
- Reverse-direction correction: `e*d < 0`; every such correction increases absolute error.
- OFF-exact shift: `e == 0` and `abs(d) > 0`; it also increases absolute error.
- No shift: `d == 0`.

The last six categories partition every stored correction. Category contributions are `mean(g * category_indicator)` over **all N crystals**, so their sum is the total OFF-minus-ON MAE change. They are not within-category mean gains. Fractions and equality tests use the exact archived float64 values. `run_summary.csv` records all category fractions and contributions for correct and shuffled pairing, including zero/tie cases.

The correction amplitude is `mean(abs(d))`; its normalized version divides by OFF MAE within the same run. Shuffled amplitude, direction, benefit, and contributions are first averaged across the 20 interventions for each crystal and then across crystals. This avoids treating 20 correlated interventions on one crystal as 20 independent observations.

## Results

{view}

Correct pairing moves the output toward the target in {100*df.correct_direction_correct.min():.2f}%–{100*df.correct_direction_correct.max():.2f}% of crystals. It reduces error relative to OFF in {100*df.correct_benefit_vs_off.min():.2f}%–{100*df.correct_benefit_vs_off.max():.2f}%. Correct target direction does not guarantee lower error: {100*df.correct_aligned_overshoot_fraction.min():.2f}%–{100*df.correct_aligned_overshoot_fraction.max():.2f}% of crystals are shifted in the correct direction but overshoot far enough to worsen absolute error. Reverse-direction corrections account for {100*df.correct_reverse_fraction.min():.2f}%–{100*df.correct_reverse_fraction.max():.2f}%.

Direction alignment is higher for correct than shuffled pairing in {aggregate['runs_direction_fraction_greater_than_shuffled']}/14 runs. For JARVIS Bulk, the shuffled mean absolute correction is {jbulk.shuffled_to_correct_abs_shift_ratio.min():.4f}–{jbulk.shuffled_to_correct_abs_shift_ratio.max():.4f} times the correct-pairing correction, whereas direction alignment falls from {100*jbulk.correct_direction_correct.min():.2f}%–{100*jbulk.correct_direction_correct.max():.2f}% to {100*jbulk.shuffled_direction_correct.min():.2f}%–{100*jbulk.shuffled_direction_correct.max():.2f}%. Thus these checkpoints exhibit an output-correction alignment advantage despite similar **mean absolute output shift**. Equal means do not establish matched per-crystal or hidden-residual magnitudes, so this remains descriptive rather than a magnitude-controlled causal direction experiment.

## Contributions to MAE improvement

Values below are in each target's stored output units. Do not average physical-unit gains across different targets.

{components}

Helpful aligned shifts contribute positively, while reverse shifts and overshoots subtract from that gain. The full run table retains tie and OFF-exact contributions, which makes the decomposition exact. The shuffled net contribution is negative in every run even though some shuffled interventions happen to improve individual crystals.

Reverse-direction corrections account for {100*df.correct_reverse_share_of_harmful_mae_loss.min():.2f}%–{100*df.correct_reverse_share_of_harmful_mae_loss.max():.2f}% of the total harmful absolute-error contribution under correct pairing. This refers to harmful error magnitude, not the fraction of harmed samples. It identifies reverse output shifts as the larger observed loss component; it does not establish a training remedy or hidden-space cause.

## Two distinct benefit fractions

The earlier report's {100*df.fraction_correct_better_than_mean_shuffled_error.min():.2f}%–{100*df.fraction_correct_better_than_mean_shuffled_error.max():.2f}% range compares **correct absolute error against each crystal's mean absolute error over 20 shuffled predictions**. The {100*df.correct_benefit_vs_off.min():.2f}%–{100*df.correct_benefit_vs_off.max():.2f}% range here compares **correct absolute error against OFF absolute error**. Both are correct for their respective comparators. Mean shuffled absolute error is not the absolute error of a mean shuffled prediction; this analysis uses the former. Neither number is an improvement fraction against an independently trained graph baseline.

## Paired descriptive bootstrap intervals

The table gives pairing MAE advantage as `100*(mean shuffled error - mean correct error)/mean shuffled error`, plus direction-fraction difference (a fraction, not percentage points). There are {BOOTSTRAP_REPLICATES} percentile bootstrap replicates per run. Each replicate samples N crystals with replacement using the same indices for correct and all shuffled predictions. All 20 stored permutations are retained; permutation×crystal pairs are never pooled as independent samples. These intervals condition on the realized 20 permutations and omit Monte Carlo uncertainty across new permutations. They describe sampling variation in the stored test set, not independent confirmatory significance tests. Shared compositions, related structures, fixed splits, and repeated training seeds can introduce dependencies the crystal bootstrap does not model. No multiplicity correction or cross-run pooled p-value is claimed.

{intervals}

Pairing MAE advantage intervals lie above zero in {aggregate['runs_pairing_mae_advantage_ci95_above_zero']}/14 runs, and direction-difference intervals lie above zero in {aggregate['runs_direction_difference_ci95_above_zero']}/14 runs. These are descriptive intervals rather than a new confirmatory verdict.

## Provenance and limits

`source_checksums.csv` records SHA256 and size for each of the 42 input NPZ/summary/sample files. `run_summary.csv` repeats each predictions NPZ checksum. The script verifies finite arrays, 20×N shuffled shape, unique sample IDs, exact category partitions, contribution sums, and consistency with the archived summary and sample CSV. Run `python analyze.py` to regenerate all outputs deterministically.

The observable `d` is the final **output difference** after the downstream graph blocks and head. It is not the hidden injected tensor and does not identify its angle, gate, or norm. Neither output alignment nor ON-versus-OFF benefit proves that correct semantics improve an independently trained graph-only model, explains the training trajectory, or licenses test-informed deployment selection. The informative next model-level control is a label-free conditional or residual-norm-matched semantic mismatch, with donor indices and actual embedding-change statistics saved.
"""
    (HERE / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
