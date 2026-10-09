"""Audit and summarize fixed frozen composition-conditioned pairing controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TASKS = (
    "Jarvis-Bandgap_MBJ", "Jarvis-Bandgap_OPT", "Jarvis-FormationEnergy",
    "Jarvis-BulkModulusKv", "MP-Bandgap", "MP-BulkModuli",
)
REPEATED = ("Jarvis-Bandgap_MBJ", "Jarvis-BulkModulusKv", "Jarvis-FormationEnergy", "MP-Bandgap")
RUNS = tuple((42, task) for task in TASKS) + tuple((seed, task) for seed in (7, 1234) for task in REPEATED)
GROUP_TYPES = ("reduced_composition", "element_set")
REQUIRED = tuple((seed, task, group) for seed, task in RUNS for group in GROUP_TYPES)
BOOTSTRAPS = 2000
FIELDS = (
    "task", "seed", "group_type", "status", "num_samples", "num_eligible",
    "coverage_fraction", "num_permutations", "num_distinct_assignments",
    "weak_coverage", "weak_coverage_reason", "correct_mae", "off_mae",
    "conditional_mean_mae", "context_global_whole_test_donor_mean_mae",
    "correct_advantage_pct_vs_conditional", "correct_advantage_pct_vs_global_context",
    "conditional_minus_correct_mean_error", "conditional_minus_correct_ci_low",
    "conditional_minus_correct_ci_high", "advantage_pct_ci_low", "advantage_pct_ci_high",
    "off_minus_correct_mean_error", "off_minus_correct_ci_low", "off_minus_correct_ci_high",
    "global_context_minus_correct_mean_error", "global_context_minus_correct_ci_low",
    "global_context_minus_correct_ci_high", "conditional_worse_than_correct_assignments",
    "fraction_samples_correct_better_than_conditional_mean", "same_embedding_fraction_mean",
    "same_description_fraction_mean", "max_rho_final_audited", "bootstrap_draws",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"invalid CSV header: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"malformed CSV row: {path}")
    return rows


def ensure(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def numeric(value, name: str) -> float:
    result = float(value)
    ensure(math.isfinite(result), f"nonfinite {name}")
    return result


def close(actual, expected, name: str, atol=1e-7):
    ensure(np.allclose(actual, expected, rtol=1e-8, atol=atol), f"inconsistent {name}")


def ratio_advantage(correct: float, alternative: float) -> float | None:
    return 100 * (1 - correct / alternative) if alternative > 0 else None


def paired_bootstrap(correct_error, conditional_error, off_error, global_error, seed: int):
    """Resample crystals, retaining all alternatives for the same sampled row.

    Conditional predictions are first averaged in absolute-error space within
    each crystal. Assignment draws are never treated as independent crystals.
    """
    n = len(correct_error)
    rng = np.random.default_rng(seed)
    ci_values = defaultdict(list)
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(0, n, size=n)
        correct = float(correct_error[sampled].mean())
        conditional = float(conditional_error[sampled].mean())
        ci_values["conditional"].append(float((conditional_error[sampled] - correct_error[sampled]).mean()))
        ci_values["off"].append(float((off_error[sampled] - correct_error[sampled]).mean()))
        ci_values["global"].append(float((global_error[sampled] - correct_error[sampled]).mean()))
        if conditional > 0:
            ci_values["advantage"].append(100 * (1 - correct / conditional))
    result = {}
    for name, values in ci_values.items():
        result[name] = [float(x) for x in np.quantile(values, (.025, .975))]
    return result


def expected_eligible(group_rows, group_type):
    counts = Counter(row[group_type] for row in group_rows)
    return np.asarray([index for index, row in enumerate(group_rows) if counts[row[group_type]] >= 2], dtype=np.int64)


def audit_parent(run_dir: Path, seed: int, task: str, group_rows):
    summary = json_read(run_dir / "summary.json")
    ensure(summary["task"] == task and int(summary["seed"]) == seed, "parent task/seed mismatch")
    ensure(summary.get("trained") is False and summary.get("checkpoint_selection") is False,
           "parent does not certify frozen checkpoint/no selection")
    for field in ("formal_reproduction_max_abs_pred_diff_native", "prior_reproduction_max_abs_pred_diff_native"):
        value = numeric(summary[field], field)
        ensure(0 <= value <= 1e-5, f"replay tolerance failed: {field}={value}")
    value = numeric(summary["formal_reproduction_max_abs_true_diff"], "truth replay")
    ensure(0 <= value <= 1e-5, "truth replay tolerance failed")
    bound = numeric(summary["max_rho_final_audited"], "parent rho")
    ensure(0 <= bound <= .1001, f"parent rho bound failed: {bound}")
    ensure(isinstance(summary.get("input_sha256"), dict) and len(summary["input_sha256"]) >= 4,
           "missing parent input hashes")
    ensure(all(len(value) == 64 and all(c in "0123456789abcdef" for c in value)
               for value in summary["input_sha256"].values()), "invalid parent input hash")
    embeddings = csv_read(run_dir / "embedding_hashes.csv")
    ensure(len(embeddings) == len(group_rows), "embedding hash count mismatch")
    for index, (embedding, group) in enumerate(zip(embeddings, group_rows)):
        ensure(int(embedding["sample_index"]) == index and embedding["sample_id"] == group["sample_id"],
               "embedding identity/order mismatch")
        value = embedding["embedding_sha256"]
        ensure(len(value) == 64 and all(c in "0123456789abcdef" for c in value), "invalid embedding hash")
    ensure({item["group_type"] for item in summary["groups"]} == set(GROUP_TYPES),
           "parent lacks both fixed group results")
    return summary, embeddings


def audit_group(root: Path, seed: int, task: str, group_type: str, group_rows, parent, embeddings):
    run_dir = root / "runs" / f"seed_{seed}" / task
    folder = run_dir / group_type
    summary = json_read(folder / "summary.json")
    ensure(summary["task"] == task and int(summary["seed"]) == seed and summary["group_type"] == group_type,
           "child task/seed/group mismatch")
    ensure(int(summary["num_samples"]) == len(group_rows), "full test count mismatch")
    eligible = expected_eligible(group_rows, group_type)
    ensure(int(summary["num_eligible"]) == len(eligible), "eligible coverage mismatch")
    parent_group = [item for item in parent["groups"] if item["group_type"] == group_type]
    ensure(len(parent_group) == 1 and parent_group[0] == summary, "parent/child summary differs")
    if not len(eligible):
        ensure(summary["status"] == "ineligible", "zero-coverage group is not ineligible")
        return {"task": task, "seed": seed, "group_type": group_type, "status": "ineligible",
                "num_samples": len(group_rows), "num_eligible": 0, "coverage_fraction": 0,
                "num_permutations": 0, "num_distinct_assignments": 0, "weak_coverage": True,
                "weak_coverage_reason": "no eligible composition group", "bootstrap_draws": 0}, {
                    "seed": seed, "task": task, "group_type": group_type, "status": "ineligible"}
    ensure(summary["status"] == "complete", "nonempty eligible group is incomplete")
    ensure(int(summary["num_permutations"]) == 20, "required 20 assignment draws absent")
    for filename in ("sample_metrics.csv", "permutation_metrics.csv", "predictions.npz"):
        ensure((folder / filename).is_file(), f"missing required {filename}")
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        required = ("truth", "correct", "off", "conditional", "global_context", "eligible_indices", "donor_indices")
        ensure(set(required).issubset(archive.files), "NPZ lacks required arrays")
        arrays = {key: archive[key].copy() for key in required}
    n = len(eligible)
    ensure(np.issubdtype(arrays["eligible_indices"].dtype, np.integer) and
           np.array_equal(arrays["eligible_indices"], eligible), "eligible index/order mismatch")
    donors = arrays["donor_indices"]
    ensure(np.issubdtype(donors.dtype, np.integer) and donors.shape == (20, n), "invalid complete donor matrix")
    keys = [row[group_type] for row in group_rows]
    for index, assigned in enumerate(donors):
        ensure(np.array_equal(np.sort(assigned), eligible), f"donor map {index} is not eligible bijection")
        ensure(np.all(assigned != eligible), f"donor map {index} has fixed point")
        ensure(all(keys[i] == keys[j] for i, j in zip(eligible, assigned)), f"donor map {index} crosses composition groups")
    for key in ("truth", "correct", "off"):
        ensure(arrays[key].shape == (n,) and np.all(np.isfinite(arrays[key])), f"invalid {key} array")
    for key in ("conditional", "global_context"):
        ensure(arrays[key].shape == (20, n) and np.all(np.isfinite(arrays[key])), f"invalid {key} matrix")
    sample = csv_read(folder / "sample_metrics.csv")
    ensure(len(sample) == n, "sample CSV row count differs")
    for index, (row, full_index) in enumerate(zip(sample, eligible)):
        ensure(int(row["sample_index"]) == full_index and row["sample_id"] == group_rows[full_index]["sample_id"]
               and row["group_key"] == keys[full_index], f"sample ID/group/order differs at row {index}")
        for column, array in (("y_true", "truth"), ("y_correct", "correct"), ("y_off", "off")):
            close(numeric(row[column], column), arrays[array][index], column)
    correct_error = np.abs(arrays["correct"] - arrays["truth"])
    off_error = np.abs(arrays["off"] - arrays["truth"])
    conditional_errors = np.abs(arrays["conditional"] - arrays["truth"][None, :])
    global_errors = np.abs(arrays["global_context"] - arrays["truth"][None, :])
    conditional_error = conditional_errors.mean(axis=0)
    global_error = global_errors.mean(axis=0)
    for column, values in (("conditional_mean_abs_error", conditional_error), ("correct_abs_error", correct_error),
                           ("global_context_mean_abs_error", global_error)):
        close(np.asarray([numeric(row[column], column) for row in sample]), values, column)
    permutations = csv_read(folder / "permutation_metrics.csv")
    ensure(len(permutations) == 20, "permutation CSV must contain 20 rows")
    same_embeddings, same_descriptions = [], []
    for index, (row, assigned) in enumerate(zip(permutations, donors)):
        ensure(int(row["permutation_index"]) == index, "permutation index/order differs")
        close(numeric(row["mae"], "permutation mae"), conditional_errors[index].mean(), "permutation mae")
        close(numeric(row["fixed_point_fraction"], "fixed points"), 0, "fixed points")
        embedding_equal = float(np.mean([embeddings[i]["embedding_sha256"] == embeddings[j]["embedding_sha256"]
                                         for i, j in zip(eligible, assigned)]))
        description_equal = float(np.mean([group_rows[i]["description_sha256"] == group_rows[j]["description_sha256"]
                                           for i, j in zip(eligible, assigned)]))
        close(numeric(row["same_embedding_fraction"], "same embedding"), embedding_equal, "same embedding rate")
        close(numeric(row["same_description_fraction"], "same description"), description_equal, "same description rate")
        same_embeddings.append(embedding_equal)
        same_descriptions.append(description_equal)
    mae_correct, mae_off = float(correct_error.mean()), float(off_error.mean())
    mae_conditional, mae_global = float(conditional_error.mean()), float(global_error.mean())
    distinct = len({assigned.tobytes() for assigned in donors})
    direct = {
        "coverage_fraction": n / len(group_rows), "correct_mae": mae_correct, "off_mae": mae_off,
        "conditional_mean_mae": mae_conditional, "context_global_whole_test_donor_mean_mae": mae_global,
        "correct_advantage_pct_vs_conditional": ratio_advantage(mae_correct, mae_conditional),
        "correct_advantage_pct_vs_global_context": ratio_advantage(mae_correct, mae_global),
        "conditional_worse_than_correct_assignments": int(np.sum(conditional_errors.mean(axis=1) > mae_correct)),
        "fraction_samples_correct_better_than_conditional_mean": float(np.mean(correct_error < conditional_error)),
        "same_embedding_fraction_mean": float(np.mean(same_embeddings)),
        "same_description_fraction_mean": float(np.mean(same_descriptions)),
    }
    ensure(int(summary["num_distinct_assignments"]) == distinct, "distinct donor count differs")
    for name, value in direct.items():
        if value is not None:
            close(numeric(summary[name], name), value, f"summary {name}")
    close(numeric(summary["conditional_mae_sd_over_assignments"], "assignment SD"),
          conditional_errors.mean(axis=1).std(ddof=1), "assignment SD")
    bound = numeric(summary["max_rho_final_audited"], "child rho")
    ensure(0 <= bound <= .1001 and bound <= float(parent["max_rho_final_audited"]) + 1e-8, "child rho bound mismatch")
    subset_replay = numeric(summary['subset_reproduction_max_abs_pred_diff_native'], 'same-path subset replay')
    ensure(0 <= subset_replay <= 1e-5, 'subset batch-path replay tolerance failed')
    for name in ('predictions.npz', 'sample_metrics.csv', 'permutation_metrics.csv'):
        ensure(sha256(folder / name) == summary.get('output_sha256', {}).get(name),
               f'completed output checksum changed: {name}')
    reason = []
    if n < 30:
        reason.append("eligible N<30")
    if distinct == 1:
        reason.append("only one distinct donor map")
    ci = paired_bootstrap(correct_error, conditional_error, off_error, global_error,
                          20261009 + seed + TASKS.index(task) * 10000 + GROUP_TYPES.index(group_type) * 100000)
    result = dict(direct, task=task, seed=seed, group_type=group_type, status="complete",
                  num_samples=len(group_rows), num_eligible=n, num_permutations=20,
                  num_distinct_assignments=distinct, weak_coverage=bool(reason), weak_coverage_reason="; ".join(reason),
                  conditional_minus_correct_mean_error=float((conditional_error - correct_error).mean()),
                  conditional_minus_correct_ci_low=ci["conditional"][0], conditional_minus_correct_ci_high=ci["conditional"][1],
                  advantage_pct_ci_low=ci.get("advantage", [None, None])[0], advantage_pct_ci_high=ci.get("advantage", [None, None])[1],
                  off_minus_correct_mean_error=float((off_error - correct_error).mean()),
                  off_minus_correct_ci_low=ci["off"][0], off_minus_correct_ci_high=ci["off"][1],
                  global_context_minus_correct_mean_error=float((global_error - correct_error).mean()),
                  global_context_minus_correct_ci_low=ci["global"][0], global_context_minus_correct_ci_high=ci["global"][1],
                  max_rho_final_audited=bound, bootstrap_draws=BOOTSTRAPS)
    audit = {"seed": seed, "task": task, "group_type": group_type, "status": "PASS",
             "eligible_order": True, "all_20_bijections": True, "no_fixed_points": True,
             "group_membership": True, "all_metrics_recomputed": True,
             "distinct_assignments": distinct, "input_hashes": {name: sha256(folder / name) for name in
                  ("summary.json", "sample_metrics.csv", "permutation_metrics.csv", "predictions.npz")},
             "donor_map_sha256": [hashlib.sha256(assigned.astype("<i8").tobytes()).hexdigest() for assigned in donors]}
    return result, audit


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seed_aggregates(rows):
    result = []
    for group_type in GROUP_TYPES:
        for seed in (42, 7, 1234):
            required_tasks = TASKS if seed == 42 else REPEATED
            subset = [row for row in rows if row["group_type"] == group_type and row["seed"] == seed]
            complete = [row for row in subset if row["status"] == "complete"]
            advantage = [row["correct_advantage_pct_vs_conditional"] for row in complete
                         if row["correct_advantage_pct_vs_conditional"] is not None]
            result.append({"group_type": group_type, "seed": seed, "required_tasks": len(required_tasks),
                           "reported_tasks": len(subset), "eligible_tasks": len(complete),
                           "ineligible_tasks": sum(row["status"] == "ineligible" for row in subset),
                           "tasks_correct_better_than_conditional": sum(value > 0 for value in advantage),
                           "tasks_correct_not_better_than_conditional": sum(value <= 0 for value in advantage),
                           "mean_task_advantage_pct": float(np.mean(advantage)) if advantage else None,
                           "weak_coverage_tasks": sum(row.get("weak_coverage", False) for row in complete),
                           "note": "mean of task-specific normalized effects; no physical-unit MAE pooling"})
    return result


def format_number(value):
    return "—" if value is None or value == "" else f"{float(value):.6g}"


def report_text(rows, aggregate, validation):
    completed = validation["status"] == "PASS"
    lines = ["# Composition-conditioned semantic pairing control", "",
             "**" + ("Complete and audited." if completed else "INCOMPLETE: progress report; no final experiment verdict.") + "**", "",
             "This exploratory analysis reuses fourteen frozen, validation-selected SFTGNN-GPSR checkpoints. "
             "Exact reduced composition is the primary donor restriction; element set is the secondary restriction. "
             "There is no training, checkpoint reselection, or outcome-based change of tasks, seeds, grouping, or the twenty draws.", "",
             f"Audit status: {validation['status']}; audited {len(rows)}/28 fixed checkpoint–group results. "
             f"Missing: {len(validation['missing'])}; audit errors: {len(validation['errors'])}.", "",
             "## Results on the same eligible subset", "",
             "The correct, OFF, conditional, and global-context MAEs in each row use exactly the same eligible crystals. "
             "Global context is sliced from the earlier whole-test global donor assignments; its donor pool differs from the conditional pool.", "",
             "| Group | Task | Seed | Eligible/full N | Coverage | Distinct maps/20 | Correct MAE | OFF MAE | Conditional MAE | Global-context MAE | Correct advantage vs conditional, % [descriptive 95% CI] | Coverage flag |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        if row["status"] == "ineligible":
            effect = "ineligible"
        else:
            effect = (f"{format_number(row['correct_advantage_pct_vs_conditional'])} "
                      f"[{format_number(row['advantage_pct_ci_low'])}, {format_number(row['advantage_pct_ci_high'])}]")
        lines.append("| " + " | ".join([row["group_type"], row["task"], str(row["seed"]),
            f"{row['num_eligible']}/{row['num_samples']}", f"{100 * row['coverage_fraction']:.2f}%",
            f"{row.get('num_distinct_assignments', 0)}/{row.get('num_permutations', 0)}",
            *[format_number(row.get(key)) for key in ("correct_mae", "off_mae", "conditional_mean_mae", "context_global_whole_test_donor_mean_mae")],
            effect, row["weak_coverage_reason"] if row["weak_coverage"] else "adequate N; scope remains subset-specific"]) + " |")
    lines.extend(["", "## Normalized summaries by grouping and training seed", "",
                  "Physical-unit MAEs are not averaged across properties. These summaries average task-specific relative effects and do not treat repeated seeds as independent tasks.", "",
                  "| Group | Seed | Audited/required tasks | Eligible tasks | Correct better | Mean task advantage (%) | Weak-coverage eligible tasks |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for item in aggregate["group_seed_summaries"]:
        lines.append(f"| {item['group_type']} | {item['seed']} | {item['reported_tasks']}/{item['required_tasks']} | "
                     f"{item['eligible_tasks']} | {item['tasks_correct_better_than_conditional']} | "
                     f"{format_number(item['mean_task_advantage_pct'])} | {item['weak_coverage_tasks']} |")
    lines.extend(["", "## Scientific interpretation and limits", "",
                  "A correct-pairing advantage under exact reduced-composition restriction is evidence that the frozen checkpoint responds to structural correspondence among the covered same-composition crystals. "
                  "Element-set restriction is chemically weaker because stoichiometry may change. A null or adverse conditional effect is retained and does not invalidate the previously observed global effect.", "",
                  "The contrast between global and composition-conditioned effects is **not a causal percentage decomposition of composition and structural information**. "
                  "Donor similarity, subgroup coverage, description duplication, and distribution shift differ. These results also do not establish gains over an independently trained graph-only model or explain CartNet's independently trained MP-bulk regression.", "",
                  "Intervals use 2,000 paired bootstrap resamples of crystals after averaging conditional absolute error within each crystal over the twenty assignment draws. "
                  "They describe this fixed checkpoint and subset. Crystals share composition groups and donor assignments, so they are statistically dependent; "
                  "these crystal-bootstrap intervals may understate composition-cluster uncertainty and are not confirmatory population-level significance tests.", "",
                  "Twenty draws can repeat the same donor map. Distinct-map counts and the preflight mapping-space sizes must accompany interpretation. "
                  "Eligible N<30 or one distinct map is explicitly marked weak coverage and retained. Groups whose descriptions or embeddings are identical can change identities without changing semantic inputs; equality rates are in task_summary.csv.", "",
                  "All available groups are reported. Full coverage, null/negative outcomes, and ineligible groups are preserved without selection by effect size. "
                  "The fixed split, limited seeds, checkpoint co-adaptation, and post-hoc use of test labels restrict generalization.", "",
                  "## Integrity", "",
                  "The auditor checks the fixed 14×2 schedule, sample identity/order, full-test donor indices, twenty eligible bijections, no fixed points, exact group membership, "
                  "description/embedding equality rates, finite predictions, CSV/NPZ/JSON metric agreement, unchanged tau bound (maximum <=0.1001), "
                  "and formal/prior replay differences <=1e-5 in native standardized output units. Input hashes and per-draw donor hashes are recorded in validation.json."])
    if validation["missing"]:
        lines.extend(["", "Pending: " + ", ".join(validation["missing"])])
    if validation["errors"]:
        lines.extend(["", "Audit errors:", "", *["- " + item for item in validation["errors"]]])
    return "\n".join(lines) + "\n"


def self_test():
    rows = [{"reduced_composition": key} for key in ("A", "A", "B", "C", "C")]
    ensure(expected_eligible(rows, "reduced_composition").tolist() == [0, 1, 3, 4], "eligibility self-test")
    result = paired_bootstrap(np.ones(4), np.ones(4) * 2, np.ones(4) * 3, np.ones(4) * 4, 1)
    close(result["conditional"], [1, 1], "paired constant difference")
    close(result["advantage"], [50, 50], "paired normalized effect")
    close(result["off"], [2, 2], "paired OFF difference")
    ensure(len(REQUIRED) == 28 and len(RUNS) == 14, "fixed schedule")
    print("FINALIZE_SELF_TEST_PASS: fixed schedule, eligibility, paired bootstrap and normalized intervals")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = args.root.resolve()
    validation = {"status": "RUNNING", "required_results": 28, "missing": [], "errors": [], "audits": [],
                  "checkpoint_replay_audits": [],
                  "bootstrap_resamples": BOOTSTRAPS, "bootstrap_unit": "paired crystal; assignment errors averaged first",
                  "interval_limit": "composition-group and donor dependence remains; descriptive crystal bootstrap only"}
    rows = []
    groups_by_task = defaultdict(list)
    parents = {}
    try:
        preflight = json_read(root / "preflight.json")
        ensure(preflight["status"] == "PASS" and preflight["num_runs"] == 14, "preflight not complete/PASS")
        ensure(preflight.get("labels_used_for_grouping") is False, "preflight grouping used labels")
        for name in ("sample_groups.csv", "group_coverage.csv"):
            ensure(sha256(root / name) == preflight["output_hashes"][name], f"preflight output hash differs: {name}")
        validation["metadata_hashes"] = {name: sha256(root / name) for name in
                                         ("preflight.json", "sample_groups.csv", "group_coverage.csv")}
        group_rows = csv_read(root / "sample_groups.csv")
        for row in group_rows:
            ensure(row["task"] in TASKS, "unexpected preflight task")
            groups_by_task[row["task"]].append(row)
        ensure(set(groups_by_task) == set(TASKS), "preflight missing fixed task")
        for task, records in groups_by_task.items():
            ensure([int(row["sample_index"]) for row in records] == list(range(len(records))), f"preflight sample order invalid: {task}")
            ensure(len({row["sample_id"] for row in records}) == len(records), f"preflight duplicate sample ID: {task}")
        for seed, task in RUNS:
            run_dir = root / "runs" / f"seed_{seed}" / task
            if not (run_dir / "summary.json").is_file() or not (run_dir / "embedding_hashes.csv").is_file():
                continue
            try:
                parents[(seed, task)] = audit_parent(run_dir, seed, task, groups_by_task[task])
                parent, _ = parents[(seed, task)]
                ensure(preflight["output_hashes"]["sample_groups.csv"] in parent["input_sha256"].values(),
                       "inference did not certify current sample_groups hash")
                validation["checkpoint_replay_audits"].append({"task": task, "seed": seed, "status": "PASS",
                    "formal_reproduction_max_abs_pred_diff_native": parent["formal_reproduction_max_abs_pred_diff_native"],
                    "prior_reproduction_max_abs_pred_diff_native": parent["prior_reproduction_max_abs_pred_diff_native"],
                    "formal_reproduction_max_abs_true_diff": parent["formal_reproduction_max_abs_true_diff"],
                    "max_rho_final_audited": parent["max_rho_final_audited"],
                    "input_sha256": parent["input_sha256"],
                    "parent_summary_sha256": sha256(run_dir / "summary.json"),
                    "embedding_hashes_sha256": sha256(run_dir / "embedding_hashes.csv")})
            except Exception as exc:
                parents.pop((seed, task), None)
                validation["errors"].append(f"seed_{seed}/{task}: {type(exc).__name__}: {exc}")
        for seed, task, group_type in REQUIRED:
            key = f"seed_{seed}/{task}/{group_type}"
            if (seed, task) not in parents or not (root / "runs" / f"seed_{seed}" / task / group_type / "summary.json").is_file():
                validation["missing"].append(key)
                continue
            try:
                parent, embeddings = parents[(seed, task)]
                result, audit = audit_group(root, seed, task, group_type, groups_by_task[task], parent, embeddings)
                rows.append(result)
                validation["audits"].append(audit)
            except Exception as exc:
                validation["errors"].append(f"{key}: {type(exc).__name__}: {exc}")
        required_folders = {f"seed_{seed}/{task}/{group_type}" for seed, task, group_type in REQUIRED}
        found = {str(path.parent.relative_to(root / "runs")).replace("\\", "/") for path in (root / "runs").glob("seed_*/*/*/summary.json")}
        ensure(not (found - required_folders), f"unexpected extra run/group results: {sorted(found - required_folders)}")
    except Exception as exc:
        validation["errors"].append(f"preflight/global: {type(exc).__name__}: {exc}")
    validation["audited_results"] = len(rows)
    validation["complete"] = len(rows) == 28 and not validation["missing"] and not validation["errors"]
    validation["status"] = "PASS" if validation["complete"] else ("FAIL" if validation["errors"] else "PARTIAL")
    aggregate = {"complete": validation["complete"], "status": validation["status"], "fixed_required_results": 28,
                 "audited_results": len(rows), "group_seed_summaries": seed_aggregates(rows),
                 "interpretation": "exploratory within-checkpoint structural correspondence sensitivity on group-eligible subsets",
                 "no_causal_composition_percentage_decomposition": True, "no_physical_mae_pooling": True,
                 "bootstrap": {"draws": BOOTSTRAPS, "unit": "paired crystal", "assignment_errors_averaged_first": True,
                               "composition_cluster_dependence_accounted_for": False}}
    root.mkdir(parents=True, exist_ok=True)
    prefix = "" if validation["complete"] else "progress_"
    write_csv(root / ("task_summary.csv" if validation["complete"] else "progress_summary.csv"), rows, FIELDS)
    (root / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / f"{prefix}aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / f"{prefix}report.md").write_text(report_text(rows, aggregate, validation), encoding="utf-8")
    print(json.dumps({"status": validation["status"], "audited": len(rows), "required": 28,
                      "missing": len(validation["missing"]), "errors": validation["errors"]}, ensure_ascii=False))
    return 0 if validation["complete"] or (args.allow_partial and not validation["errors"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
