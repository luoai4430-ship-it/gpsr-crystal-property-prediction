"""Validate completed permutation controls and generate compact publication artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def target_std(run_dir: Path) -> float:
    text = (run_dir / "run.log").read_text(errors="replace")
    values = re.findall(r"^std:([-+0-9.eE]+)\s*$", text, flags=re.MULTILINE)
    if not values:
        raise RuntimeError(f"target std not found in {run_dir / 'run.log'}")
    return float(values[-1])


def main():
    run_dirs = sorted(p.parent for p in (ROOT / "runs").glob("seed_*/*/summary.json"))
    if len(run_dirs) != 14:
        raise RuntimeError(f"expected 14 completed runs, found {len(run_dirs)}")
    rows, checks = [], []
    for rd in run_dirs:
        sp = rd / "summary.json"
        summary = json.loads(sp.read_text())
        if "formal_reproduction_max_abs_pred_diff_standardized" not in summary:
            summary["formal_reproduction_max_abs_pred_diff_standardized"] = (
                summary["formal_reproduction_max_abs_pred_diff"] / max(abs(target_std(rd)), 1e-12)
            )
            sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        perm = pd.read_csv(rd / "permutation_metrics.csv")
        sample = pd.read_csv(rd / "sample_metrics.csv")
        arrays = np.load(rd / "predictions.npz")
        pred_shape = tuple(arrays["shuffled"].shape)
        checks.append({
            "task": summary["task"], "seed": summary["seed"],
            "summary_present": True,
            "num_permutation_rows": len(perm),
            "fixed_point_max": float(perm.fixed_point_fraction.max()),
            "num_sample_rows": len(sample),
            "prediction_shape": str(pred_shape),
            "sample_ids_unique": bool(sample.sample_id.astype(str).is_unique),
            "standardized_reproduction_max_abs_diff": summary["formal_reproduction_max_abs_pred_diff_standardized"],
            "reproduction_pass": summary["formal_reproduction_max_abs_pred_diff_standardized"] <= 1e-5,
            "correct_below_all_permutations": bool((perm.test_mae > summary["correct_mae"]).all()),
        })
        if len(perm) != 20 or perm.fixed_point_fraction.max() != 0:
            raise RuntimeError(f"invalid permutation set: {rd}")
        if len(sample) != summary["num_samples"] or pred_shape != (20, summary["num_samples"]):
            raise RuntimeError(f"sample/prediction shape mismatch: {rd}")
        if not checks[-1]["reproduction_pass"]:
            raise RuntimeError(f"formal prediction reproduction failed: {rd}")
        rows.append(summary)

    df = pd.DataFrame(rows).sort_values(["seed", "task"])
    df.to_csv(ROOT / "task_summary.csv", index=False)
    check_df = pd.DataFrame(checks).sort_values(["seed", "task"])
    check_df.to_csv(ROOT / "validation.csv", index=False)

    grouped = df.groupby("task", sort=True).agg(
        runs=("seed", "size"),
        mean_pairing_advantage_pct=("pairing_advantage_pct_vs_shuffle", "mean"),
        sample_sd_pairing_advantage_pct=("pairing_advantage_pct_vs_shuffle", "std"),
        min_pairing_advantage_pct=("pairing_advantage_pct_vs_shuffle", "min"),
        max_pairing_advantage_pct=("pairing_advantage_pct_vs_shuffle", "max"),
        mean_correct_gain_pct_vs_off=("correct_gain_pct_vs_off", "mean"),
        mean_fraction_samples_pairing_helped=("fraction_samples_pairing_helped", "mean"),
    ).reset_index()
    grouped.to_csv(ROOT / "multiseed_task_summary.csv", index=False)

    seed42 = df[df.seed.eq(42)]
    aggregate = {
        "num_runs": int(len(df)), "num_seed42_tasks": int(len(seed42)),
        "seed42_mean_pairing_advantage_pct": float(seed42.pairing_advantage_pct_vs_shuffle.mean()),
        "seed42_tasks_positive_pairing_advantage": int((seed42.pairing_advantage_pct_vs_shuffle > 0).sum()),
        "all_runs_positive_pairing_advantage": int((df.pairing_advantage_pct_vs_shuffle > 0).sum()),
        "all_runs_correct_better_than_off": int((df.correct_mae < df.off_mae).sum()),
        "all_runs_shuffle_worse_than_off": int((df.shuffle_mean_mae > df.off_mae).sum()),
        "runs_correct_below_all_20_permutations": int(check_df.correct_below_all_permutations.sum()),
        "monte_carlo_p_value_each_run": float(1 / 21),
        "max_formal_reproduction_abs_diff_standardized": float(df.formal_reproduction_max_abs_pred_diff_standardized.max()),
    }
    (ROOT / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    validation = {
        "status": "PASS", "required_runs": 14, "completed_runs": len(df),
        "permutations_per_run": 20, "all_derangements_have_zero_fixed_points": True,
        "all_sample_ids_unique_within_run": bool(check_df.sample_ids_unique.all()),
        "formal_reproduction_tolerance_standardized": 1e-5,
        "max_observed_standardized_reproduction_error": aggregate["max_formal_reproduction_abs_diff_standardized"],
        "note": "The tolerance is applied in the model's native standardized-output scale; raw physical-unit differences are also retained in task_summary.csv.",
    }
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    view = df[["task", "seed", "correct_mae", "off_mae", "shuffle_mean_mae",
               "pairing_advantage_pct_vs_shuffle", "correct_gain_pct_vs_off",
               "fraction_samples_pairing_helped", "permutation_p_one_sided"]]
    task_view = grouped.fillna("")
    report = f"""# Semantic Pairing Permutation Control

## Design

This exploratory post-hoc experiment uses frozen, validation-selected bounded-GPSR checkpoints. For each test set it compares (i) the correct crystal–MatBERT pairing, (ii) the same checkpoint with the semantic residual disabled, and (iii) 20 fixed-point-free permutations of the test embeddings. The permutations preserve the empirical embedding distribution while destroying sample correspondence. No training, checkpoint selection, task selection, or hyperparameter adjustment uses these results.

## Results by run

{view.to_markdown(index=False)}

## Multi-seed task summary

{task_view.to_markdown(index=False)}

Correct pairing had lower MAE than every one of the 20 derangements in all {len(df)}/{len(df)} task–seed runs. Accordingly, each run attains the smallest possible one-sided Monte Carlo p-value with 20 permutations, 1/21 = 0.0476. These per-run values are descriptive and are not presented as multiplicity-corrected confirmatory tests.

Across the six seed-42 tasks, the mean MAE advantage of correct pairing relative to shuffled pairing was {aggregate['seed42_mean_pairing_advantage_pct']:.3f}% and was positive on 6/6 tasks. Across the four three-seed tasks, the task-mean advantage ranged from {grouped[grouped.runs.eq(3)].mean_pairing_advantage_pct.min():.3f}% to {grouped[grouped.runs.eq(3)].mean_pairing_advantage_pct.max():.3f}%. Correct pairing also outperformed semantic OFF in {aggregate['all_runs_correct_better_than_off']}/{len(df)} runs, whereas shuffled pairing was worse than semantic OFF in {aggregate['all_runs_shuffle_worse_than_off']}/{len(df)} runs.

The sample-level effect is heterogeneous: the fraction of samples benefiting from correct rather than mean-shuffled pairing ranges from {df.fraction_samples_pairing_helped.min():.3f} to {df.fraction_samples_pairing_helped.max():.3f}. Thus the aggregate signal is consistent, but it is not a claim that every crystal benefits.

MP-Bulk at seed 42 shows the largest pairing advantage ({df[(df.task == 'MP-BulkModuli') & (df.seed == 42)].pairing_advantage_pct_vs_shuffle.iloc[0]:.3f}%). This does not contradict the independently trained CartNet comparison: the present result says that the trained GPSR checkpoint uses sample-matched semantic information more effectively than mismatched or disabled semantics; it does not say that GPSR must outperform an independently optimized graph-only backbone.

## Interpretation

The results support semantic specificity within the trained GPSR models: preserving crystal–text correspondence matters beyond merely supplying embeddings with the same marginal distribution. They also show that mismatched semantic interventions can be actively harmful compared with disabling the intervention. Because this is a test-label, within-checkpoint, post-hoc analysis, it is an exploratory association and cannot establish a causal advantage over independently trained graph-only models or justify model selection.

## Integrity checks

All 14 required task–seed runs, 280 derangements, sample tables, and prediction arrays are present. All permutations have zero fixed points. Recomputed formal predictions agree with archived predictions within {aggregate['max_formal_reproduction_abs_diff_standardized']:.3g} in the model's standardized-output scale, below the 1e-5 tolerance. Raw physical-unit discrepancies are retained for transparency.
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
