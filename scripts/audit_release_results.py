#!/usr/bin/env python3
"""Audit and optionally refresh compact GPSR release summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import zlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments/final_10task/summary/multiseed_paired_results.csv"
ALIGNN = ROOT / "experiments/backbone_transfer_alignn/summary/alignn_multiseed_paired.csv"
PAIRING = ROOT / "experiments/semantic_pairing_permutation/task_summary.csv"
OUTPUT_CORRECTION = ROOT / "experiments/semantic_pairing_permutation/output_correction_analysis"
COMPOSITION_PAIRING = ROOT / "experiments/composition_pairing_control"


def semantic_pairing_scope():
    seed42 = {"Jarvis-Bandgap_MBJ", "Jarvis-Bandgap_OPT", "Jarvis-FormationEnergy",
              "Jarvis-BulkModulusKv", "MP-Bandgap", "MP-BulkModuli"}
    repeated = seed42 - {"Jarvis-Bandgap_OPT", "MP-BulkModuli"}
    return {(task, 42) for task in seed42} | {(task, seed) for task in repeated for seed in (7, 1234)}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_sd(values):
    return statistics.stdev(values)


def write_csv(path: Path, fieldnames, data):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def audit_sft(write: bool):
    data = rows(SFT)
    assert len(data) == 30, f"expected 30 SFTGNN rows, found {len(data)}"
    assert {int(row["seed"]) for row in data} == {7, 42, 1234}
    improved = sum(float(row["final_mae"]) < float(row["graph_mae"]) for row in data)
    assert improved == 28, f"expected 28/30 SFTGNN improvements, found {improved}/30"

    by_seed = defaultdict(list)
    by_task = defaultdict(list)
    for row in data:
        by_seed[int(row["seed"])].append(float(row["paired_ratio_final_over_graph"]))
        by_task[row["task"]].append(row)
    macro = [statistics.mean(by_seed[seed]) for seed in sorted(by_seed)]
    macro_mean = statistics.mean(macro)
    macro_sd = sample_sd(macro)
    assert abs(macro_mean - 0.9663723472191945) < 1e-12
    assert abs(macro_sd - 0.009000367058585897) < 1e-12

    if write:
        task_rows = []
        for task, group in by_task.items():
            graph = [float(row["graph_mae"]) for row in group]
            final = [float(row["final_mae"]) for row in group]
            ratio = [float(row["paired_ratio_final_over_graph"]) for row in group]
            task_rows.append({
                "task": task,
                "graph_mean_mae": statistics.mean(graph),
                "graph_std_mae": sample_sd(graph),
                "final_mean_mae": statistics.mean(final),
                "final_std_mae": sample_sd(final),
                "mean_paired_ratio": statistics.mean(ratio),
                "std_paired_ratio": sample_sd(ratio),
                "relative_gain_pct": (1 - statistics.mean(ratio)) * 100,
                "improved_seeds_count": sum(float(row["final_mae"]) < float(row["graph_mae"]) for row in group),
                "num_seeds": len(group),
            })
        task_path = ROOT / "experiments/final_10task/summary/multiseed_task_summary.csv"
        write_csv(task_path, list(task_rows[0]), task_rows)

        macro_path = ROOT / "experiments/final_10task/summary/multiseed_macro_summary.csv"
        old = rows(macro_path)[0]
        old["macro_normalized_mae_mean"] = repr(macro_mean)
        old["macro_normalized_mae_std"] = repr(macro_sd)
        write_csv(macro_path, list(old), [old])

        for json_path in [
            ROOT / "experiments/final_10task/completion_status.json",
            ROOT / "experiments/final_10task/summary/verdict.json",
        ]:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
            if "verdict" in obj and isinstance(obj["verdict"], dict):
                obj["verdict"]["multiseed_macro_std"] = macro_sd
            elif "summary" in obj:
                obj["summary"]["multiseed_macro_std"] = macro_sd
            else:
                obj["multiseed_macro_std"] = macro_sd
            json_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

        three = ROOT / "experiments/cartnet_gpsr/summary/three_backbone_summary.csv"
        text = three.read_text(encoding="utf-8").replace("29/30", "28/30")
        three.write_text(text, encoding="utf-8")
        report = ROOT / "experiments/cartnet_gpsr/summary/cartnet_gpsr_report.md"
        report.write_text(report.read_text(encoding="utf-8").replace("29/30", "28/30"), encoding="utf-8")

    return macro_mean, macro_sd, improved


def audit_alignn(write: bool):
    data = rows(ALIGNN)
    assert len(data) == 30, f"expected 30 ALIGNN rows, found {len(data)}"
    by_seed = defaultdict(list)
    by_task = defaultdict(list)
    for row in data:
        by_seed[int(row["seed"])].append(float(row["ratio"]))
        by_task[row["task"]].append(row)
    macro = [statistics.mean(by_seed[seed]) for seed in sorted(by_seed)]
    macro_mean = statistics.mean(macro)
    macro_sd = sample_sd(macro)
    improved = sum(float(row["hard_mae"]) < float(row["graph_mae"]) for row in data)
    assert improved == 26
    assert abs(macro_mean - 0.9702851787687328) < 1e-12
    assert abs(macro_sd - 0.00658507226600193) < 1e-12

    if write:
        out = []
        for task, group in by_task.items():
            graph = [float(row["graph_mae"]) for row in group]
            hard = [float(row["hard_mae"]) for row in group]
            ratio = [float(row["ratio"]) for row in group]
            out.append({
                "task": task,
                "graph_mean_mae": statistics.mean(graph),
                "graph_std_mae": sample_sd(graph),
                "hard_mean_mae": statistics.mean(hard),
                "hard_std_mae": sample_sd(hard),
                "mean_paired_ratio": statistics.mean(ratio),
                "mean_gain_pct": (1 - statistics.mean(ratio)) * 100,
                "improved_seeds_count": sum(float(row["hard_mae"]) < float(row["graph_mae"]) for row in group),
                "num_seeds": len(group),
            })
        path = ROOT / "experiments/backbone_transfer_alignn/summary/alignn_multiseed_summary.csv"
        write_csv(path, list(out[0]), out)
        report = ROOT / "experiments/backbone_transfer_alignn/summary/alignn_transfer_report.md"
        text = report.read_text(encoding="utf-8").split("## Phase C", 1)[0].rstrip()
        table = [
            "| task | graph mean MAE | graph sample SD | GPSR mean MAE | GPSR sample SD | mean GN-MAE | mean gain (%) | improved seeds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in out:
            table.append(
                f"| {row['task']} | {row['graph_mean_mae']:.8g} | {row['graph_std_mae']:.8g} | "
                f"{row['hard_mean_mae']:.8g} | {row['hard_std_mae']:.8g} | "
                f"{row['mean_paired_ratio']:.8g} | {row['mean_gain_pct']:.6g} | "
                f"{row['improved_seeds_count']}/{row['num_seeds']} |"
            )
        phase_c = [
            "## Phase C",
            f"3-seed Macro NMAE={macro_mean:.6f} ± {macro_sd:.6f}; gain={(1-macro_mean)*100:.2f}%.",
            "Tasks improved in at least 2/3 seeds: 9/10.",
            "",
            *table,
            "",
            "**Final verdict: BACKBONE_AGNOSTIC_GRAPH_PRESERVING_GAIN**",
            "",
        ]
        report.write_text(text + "\n\n" + "\n".join(phase_c), encoding="utf-8")

    return macro_mean, macro_sd, improved


def audit_semantic_pairing():
    data = rows(PAIRING)
    assert len(data) == 14, f"expected 14 semantic-pairing rows, found {len(data)}"
    pairs = {(row["task"], int(row["seed"])) for row in data}
    assert pairs == semantic_pairing_scope()
    assert sum(float(row["pairing_advantage_pct_vs_shuffle"]) > 0 for row in data) == 14
    assert sum(float(row["correct_mae"]) < float(row["off_mae"]) for row in data) == 14
    assert sum(float(row["shuffle_mean_mae"]) > float(row["off_mae"]) for row in data) == 14
    assert all(int(row["num_permutations"]) == 20 for row in data)
    assert all(float(row["formal_reproduction_max_abs_pred_diff_standardized"]) <= 1e-5
               for row in data)
    seed42 = [float(row["pairing_advantage_pct_vs_shuffle"]) for row in data
              if int(row["seed"]) == 42]
    assert len(seed42) == 6
    seed42_mean = statistics.mean(seed42)
    assert abs(seed42_mean - 21.532316345442116) < 1e-12
    return seed42_mean, len(data)


def audit_output_correction():
    """Audit compact mathematical identities; no NPZ files or model execution.

    Stored bootstrap intervals cannot be regenerated without raw predictions.
    Their finite/domain/order checks here are not an independent validation of
    the bootstrap procedure; the distributed analysis script documents that.
    """
    data = rows(OUTPUT_CORRECTION / "run_summary.csv")
    pairing = {(row["task"], int(row["seed"])): row for row in rows(PAIRING)}
    expected = semantic_pairing_scope()
    keys = [(row["task"], int(row["seed"])) for row in data]
    assert len(data) == 14 and len(set(keys)) == 14 and set(keys) == expected
    assert set(pairing) == expected
    groups = ("aligned_helpful", "aligned_overshoot", "aligned_tie", "reverse",
              "off_exact_shift", "no_shift")

    def close(actual, expected_value, label):
        actual, expected_value = float(actual), float(expected_value)
        assert math.isfinite(actual) and math.isfinite(expected_value), label
        assert math.isclose(actual, expected_value, rel_tol=1e-10, abs_tol=1e-12), (
            f"output correction mismatch: {label}: {actual} != {expected_value}")

    def number(row, name):
        value = float(row[name])
        assert math.isfinite(value), f"nonfinite output correction: {name}"
        return value

    for key, row in zip(keys, data):
        prior = pairing[key]
        assert int(row["num_permutations"]) == int(prior["num_permutations"]) == 20
        assert int(row["num_samples"]) == int(prior["num_samples"]) > 1
        assert re.fullmatch(r"[0-9a-f]{64}", row["source_predictions_sha256"])
        close(row["off_mae"], prior["off_mae"], f"{key} OFF")
        close(row["correct_abs_error"], prior["correct_mae"], f"{key} correct")
        close(row["shuffled_abs_error"], prior["shuffle_mean_mae"], f"{key} shuffled")
        close(row["pairing_advantage_pct_vs_shuffled"], prior["pairing_advantage_pct_vs_shuffle"], f"{key} pairing")
        close(row["fraction_correct_better_than_mean_shuffled_error"], prior["fraction_samples_pairing_helped"], f"{key} benefit comparator")
        off = number(row, "off_mae")
        assert off > 0
        for prefix in ("correct", "shuffled"):
            metrics = {name: number(row, prefix + "_" + name) for name in (
                "abs_error", "abs_shift", "direction_correct", "benefit_vs_off",
                "harm_vs_off", "tie_vs_off", "mae_gain_vs_off", "positive_mae_gain",
                "harmful_mae_loss", "shift_square", "error_shift_product")}
            assert metrics["abs_error"] >= 0 and metrics["abs_shift"] > 0
            assert metrics["positive_mae_gain"] >= 0 and metrics["harmful_mae_loss"] > 0
            assert metrics["shift_square"] >= metrics["abs_shift"] ** 2 - 1e-12
            for name in ("direction_correct", "benefit_vs_off", "harm_vs_off", "tie_vs_off"):
                assert 0 <= metrics[name] <= 1, (key, prefix, name)
            fractions = {g: number(row, prefix + "_" + g + "_fraction") for g in groups}
            contribution = {g: number(row, prefix + "_" + g + "_mae_gain_contribution") for g in groups}
            assert all(0 <= value <= 1 for value in fractions.values())
            close(sum(fractions.values()), 1, f"{key} {prefix} category partition")
            close(sum(contribution.values()), metrics["mae_gain_vs_off"], f"{key} {prefix} contributions")
            close(metrics["mae_gain_vs_off"], off - metrics["abs_error"], f"{key} {prefix} MAE gain")
            close(metrics["positive_mae_gain"] - metrics["harmful_mae_loss"], metrics["mae_gain_vs_off"], f"{key} {prefix} net gain")
            close(contribution["aligned_helpful"], metrics["positive_mae_gain"], f"{key} {prefix} positive contribution")
            close(-sum(contribution[g] for g in ("aligned_overshoot", "reverse", "off_exact_shift")), metrics["harmful_mae_loss"], f"{key} {prefix} loss contributions")
            assert contribution["aligned_helpful"] >= 0
            assert all(contribution[g] <= 0 for g in ("aligned_overshoot", "reverse", "off_exact_shift"))
            close(contribution["aligned_tie"], 0, f"{key} {prefix} tie")
            close(contribution["no_shift"], 0, f"{key} {prefix} no shift")
            close(metrics["direction_correct"], sum(fractions[g] for g in ("aligned_helpful", "aligned_overshoot", "aligned_tie")), f"{key} {prefix} direction")
            close(metrics["benefit_vs_off"], fractions["aligned_helpful"], f"{key} {prefix} benefit")
            close(metrics["harm_vs_off"], sum(fractions[g] for g in ("aligned_overshoot", "reverse", "off_exact_shift")), f"{key} {prefix} harm")
            close(metrics["tie_vs_off"], fractions["aligned_tie"] + fractions["no_shift"], f"{key} {prefix} tie fraction")
            close(metrics["benefit_vs_off"] + metrics["harm_vs_off"] + metrics["tie_vs_off"], 1, f"{key} {prefix} outcome partition")
            close(row[prefix + "_abs_shift_over_off_mae"], metrics["abs_shift"] / off, f"{key} {prefix} amplitude")
            close(row[prefix + "_gain_pct_vs_off"], 100 * metrics["mae_gain_vs_off"] / off, f"{key} {prefix} percentage")
            close(row[prefix + "_reverse_share_of_harmful_mae_loss"], -contribution["reverse"] / metrics["harmful_mae_loss"], f"{key} {prefix} reverse share")
        close(row["shuffled_to_correct_abs_shift_ratio"], number(row, "shuffled_abs_shift") / number(row, "correct_abs_shift"), f"{key} amplitude ratio")
        advantage = number(row, "shuffled_abs_error") - number(row, "correct_abs_error")
        close(row["pairing_mae_advantage"], advantage, f"{key} MAE advantage")
        close(row["pairing_advantage_pct_vs_shuffled"], 100 * advantage / number(row, "shuffled_abs_error"), f"{key} ratio")
        differences = {name + "_difference": ("correct_" + name, "shuffled_" + name)
                       for name in ("direction_correct", "abs_shift", "benefit_vs_off")}
        differences.update(overshoot_difference=("correct_aligned_overshoot_fraction", "shuffled_aligned_overshoot_fraction"),
                           reverse_difference=("correct_reverse_fraction", "shuffled_reverse_fraction"))
        for name, (correct, shuffled) in differences.items():
            close(row[name], number(row, correct) - number(row, shuffled), f"{key} {name}")
        for name in ("pairing_mae_advantage", "pairing_advantage_pct", *differences):
            low, high = number(row, name + "_ci95_low"), number(row, name + "_ci95_high")
            assert low <= high, (key, name, "inverted CI")
            if name not in ("pairing_mae_advantage", "pairing_advantage_pct", "abs_shift_difference"):
                assert -1 <= low <= high <= 1, (key, name, "fraction-difference CI domain")
            if name == "pairing_advantage_pct":
                assert high <= 100 + 1e-12, (key, "relative advantage CI domain")
        assert int(row["bootstrap_seed"]) == 20261009 + key[1] + zlib.adler32(key[0].encode())

    task_data = rows(OUTPUT_CORRECTION / "task_summary.csv")
    assert len(task_data) == 6 and {row["task"] for row in task_data} == {key[0] for key in expected}
    for task_row in task_data:
        group = [row for row in data if row["task"] == task_row["task"]]
        assert int(task_row["num_runs"]) == len(group)
        assert task_row["seeds"] == "/".join(str(seed) for seed in sorted(int(row["seed"]) for row in group))
        for name in task_row:
            if name.startswith("mean_"):
                metric = name[5:]
                values = [number(row, metric) for row in group]
                close(task_row[name], statistics.mean(values), f"{task_row['task']} {name}")
                sd_key = "sample_sd_" + metric
                if len(values) > 1:
                    close(task_row[sd_key], statistics.stdev(values), f"{task_row['task']} {sd_key}")
                else:
                    assert task_row[sd_key] == "", (task_row["task"], sd_key)

    sources = rows(OUTPUT_CORRECTION / "source_checksums.csv")
    expected_sources = {f"runs/seed_{seed}/{task}/{name}" for task, seed in expected
                        for name in ("predictions.npz", "summary.json", "sample_metrics.csv")}
    assert len(sources) == 42 and {row["source_file"] for row in sources} == expected_sources
    source_by_path = {row["source_file"]: row for row in sources}
    for source in sources:
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        assert int(source["bytes"]) > 0
        if not source["source_file"].endswith("predictions.npz"):
            compact = OUTPUT_CORRECTION.parent / source["source_file"].replace("runs/", "run_summaries/", 1)
            assert compact.is_file(), f"missing compact source: {compact}"
            assert hashlib.sha256(compact.read_bytes()).hexdigest() == source["sha256"]
            assert compact.stat().st_size == int(source["bytes"])
    for key, row in zip(keys, data):
        path = f"runs/seed_{key[1]}/{key[0]}/predictions.npz"
        assert row["source_predictions_sha256"] == source_by_path[path]["sha256"]

    aggregate = json.loads((OUTPUT_CORRECTION / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["backbone"] == "SFTGNN"
    assert aggregate["model_inference"] is False and aggregate["training"] is False
    assert aggregate["num_runs"] == 14 and aggregate["num_tasks_seed42"] == 6
    assert aggregate["permutations_per_run"] == 20
    assert aggregate["analysis_script_sha256"] == hashlib.sha256((OUTPUT_CORRECTION / "analyze.py").read_bytes()).hexdigest()
    assert aggregate["bootstrap"]["replicates"] == 2000
    assert aggregate["bootstrap"]["base_seed"] == 20261009
    assert aggregate["bootstrap"]["unit"] == "test crystal"
    ranges = {
        "correct_vs_off_benefit_fraction_range": "correct_benefit_vs_off",
        "correct_vs_mean_shuffled_benefit_fraction_range": "fraction_correct_better_than_mean_shuffled_error",
        "correct_direction_fraction_range": "correct_direction_correct",
        "correct_aligned_overshoot_fraction_range": "correct_aligned_overshoot_fraction",
        "correct_reverse_fraction_range": "correct_reverse_fraction",
        "correct_reverse_share_of_harmful_mae_loss_range": "correct_reverse_share_of_harmful_mae_loss",
    }
    for name, metric in ranges.items():
        values = [number(row, metric) for row in data]
        assert len(aggregate[name]) == 2
        close(aggregate[name][0], min(values), name + " min")
        close(aggregate[name][1], max(values), name + " max")
    for name, metric in {
        "runs_direction_fraction_greater_than_shuffled": "direction_correct_difference",
        "runs_direction_difference_ci95_above_zero": "direction_correct_difference_ci95_low",
        "runs_pairing_mae_advantage_ci95_above_zero": "pairing_mae_advantage_ci95_low",
    }.items():
        assert aggregate[name] == sum(number(row, metric) > 0 for row in data)
    macro = statistics.mean(number(row, "pairing_advantage_pct_vs_shuffled") for row in data if int(row["seed"]) == 42)
    close(aggregate["seed42_macro_pairing_advantage_pct"], macro, "seed42 macro")
    return len(data), aggregate["correct_vs_off_benefit_fraction_range"]


def audit_composition_pairing():
    """Audit only distributed composition tables/attestations, never raw NPZ.

    Donor membership, sample ordering and bootstrap estimates are attested by
    the full-data validation. This compact audit checks their consistency with
    distributed summaries and hashes; it cannot repeat raw-data validation.
    Null and negative outcomes are valid and receive the same checks.
    """
    folder = COMPOSITION_PAIRING
    pair_scope = semantic_pairing_scope()
    group_types = ("reduced_composition", "element_set")
    expected = {(task, seed, group) for task, seed in pair_scope for group in group_types}
    data = rows(folder / "task_summary.csv")
    keys = [(row["task"], int(row["seed"]), row["group_type"]) for row in data]
    assert len(data) == len(set(keys)) == 28 and set(keys) == expected
    pairing = {(row["task"], int(row["seed"])): row for row in rows(PAIRING)}
    assert set(pairing) == pair_scope

    def json_file(path):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        def finite_tree(item):
            if isinstance(item, float):
                assert math.isfinite(item), f"nonfinite composition JSON: {path}"
            elif isinstance(item, dict):
                for child in item.values():
                    finite_tree(child)
            elif isinstance(item, list):
                for child in item:
                    finite_tree(child)
        finite_tree(value)
        return value

    def number(value, label):
        result = float(value)
        assert math.isfinite(result), f"nonfinite composition value: {label}"
        return result

    def close(actual, value, label):
        actual, value = number(actual, label), number(value, label)
        assert math.isclose(actual, value, rel_tol=1e-8, abs_tol=1e-10), (
            f"composition mismatch: {label}: {actual} != {value}")

    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def hashes(mapping, label):
        assert isinstance(mapping, dict) and mapping, label
        assert all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in mapping.values()), label

    def boolean(value):
        assert value in (True, False, "True", "False", "true", "false"), value
        return value is True or value in ("True", "true")

    preflight = json_file(folder / "preflight.json")
    assert preflight["status"] == "PASS" and preflight["num_runs"] == 14
    assert preflight["labels_used_for_grouping"] is False
    assert preflight["num_permutations_each"] == 20
    assert preflight["primary_group_type"] == group_types[0]
    assert preflight["secondary_group_type"] == group_types[1]
    assert {(item["task"], int(item["seed"])) for item in preflight["runs"]} == pair_scope
    assert len(preflight["runs"]) == 14
    hashes(preflight["output_hashes"], "preflight output hashes")
    assert preflight["output_hashes"]["group_coverage.csv"] == file_hash(folder / "group_coverage.csv")
    coverage_rows = rows(folder / "group_coverage.csv")
    coverage = {(row["task"], row["group_type"]): row for row in coverage_rows}
    assert len(coverage_rows) == len(coverage) == 12
    assert set(coverage) == {(task, group) for task, _ in pair_scope for group in group_types}
    for (task, group), row in coverage.items():
        n, eligible = int(row["num_samples"]), int(row["num_eligible"])
        assert n > 0 and 0 <= eligible <= n
        assert n == int(pairing[(task, 42)]["num_samples"])
        close(row["coverage_fraction"], eligible / n, f"{task}/{group} coverage")
        assert int(row["num_groups_eligible"]) >= 0
        assert (eligible == 0) == (int(row["num_groups_eligible"]) == 0)
        if eligible:
            assert int(row["distinct_mapping_space_capped"]) >= 1
        else:
            assert int(row["distinct_mapping_space_capped"]) == 0

    validation = json_file(folder / "validation.json")
    assert validation["status"] == "PASS" and validation["complete"] is True
    assert validation["required_results"] == validation["audited_results"] == 28
    assert not validation["missing"] and not validation["errors"]
    assert validation["bootstrap_resamples"] == 2000
    for filename in ("preflight.json", "group_coverage.csv"):
        assert validation["metadata_hashes"][filename] == file_hash(folder / filename)
    replays = {(item["task"], int(item["seed"])): item for item in validation["checkpoint_replay_audits"]}
    assert len(validation["checkpoint_replay_audits"]) == len(replays) == 14 and set(replays) == pair_scope
    audits = {(item["task"], int(item["seed"]), item["group_type"]): item for item in validation["audits"]}
    assert len(validation["audits"]) == len(audits) == 28 and set(audits) == expected
    parents = {}
    for task, seed in pair_scope:
        path = folder / "run_summaries" / f"seed_{seed}" / task / "summary.json"
        parent = json_file(path)
        assert parent["task"] == task and int(parent["seed"]) == seed
        assert parent["trained"] is False and parent["checkpoint_selection"] is False
        hashes(parent["input_sha256"], "parent input provenance")
        assert len(parent["input_sha256"]) >= 4
        assert preflight["output_hashes"]["sample_groups.csv"] in parent["input_sha256"].values()
        replay = replays[(task, seed)]
        assert replay["status"] == "PASS"
        assert replay["parent_summary_sha256"] == file_hash(path)
        assert replay["input_sha256"] == parent["input_sha256"]
        for field in ("formal_reproduction_max_abs_pred_diff_native",
                      "prior_reproduction_max_abs_pred_diff_native", "formal_reproduction_max_abs_true_diff"):
            value = number(parent[field], field)
            assert 0 <= value <= 1e-5, (task, seed, field)
            close(replay[field], value, f"{task}/{seed} replay {field}")
        rho = number(parent["max_rho_final_audited"], "parent rho")
        assert 0 <= rho <= .1001
        close(replay["max_rho_final_audited"], rho, f"{task}/{seed} replay rho")
        assert number(parent["target_std"], "target std") > 0
        assert len(parent["groups"]) == 2 and {item["group_type"] for item in parent["groups"]} == set(group_types)
        parents[(task, seed)] = parent

    numerical_columns = set(data[0]) - {"task", "seed", "group_type", "status", "weak_coverage", "weak_coverage_reason"}
    for key, row in zip(keys, data):
        task, seed, group = key
        for field in numerical_columns:
            if row[field] != "":
                number(row[field], f"{key} {field}")
        metadata = coverage[(task, group)]
        n, eligible = int(row["num_samples"]), int(row["num_eligible"])
        assert n == int(metadata["num_samples"]) == int(pairing[(task, seed)]["num_samples"])
        assert eligible == int(metadata["num_eligible"])
        close(row["coverage_fraction"], eligible / n, f"{key} coverage")
        run = folder / "run_summaries" / f"seed_{seed}" / task
        path = run / group / "summary.json"
        child = json_file(path)
        parent = parents[(task, seed)]
        assert child == next(item for item in parent["groups"] if item["group_type"] == group)
        assert (child["task"], int(child["seed"]), child["group_type"]) == key
        assert child["num_samples"] == n and child["num_eligible"] == eligible
        audit = audits[key]
        if eligible == 0:
            assert row["status"] == child["status"] == audit["status"] == "ineligible"
            assert int(row["num_permutations"]) == int(row["num_distinct_assignments"]) == int(row["bootstrap_draws"]) == 0
            assert boolean(row["weak_coverage"])
            assert row["weak_coverage_reason"] == "no eligible composition group"
            continue
        assert row["status"] == child["status"] == "complete" and audit["status"] == "PASS"
        assert int(row["num_permutations"]) == child["num_permutations"] == 20
        assert int(row["bootstrap_draws"]) == 2000
        distinct = int(row["num_distinct_assignments"])
        assert 1 <= distinct <= 20 and distinct == child["num_distinct_assignments"] == audit["distinct_assignments"]
        assert distinct <= int(metadata["distinct_mapping_space_capped"])
        reasons = (["eligible N<30"] if eligible < 30 else []) + (["only one distinct donor map"] if distinct == 1 else [])
        assert boolean(row["weak_coverage"]) == bool(reasons)
        assert row["weak_coverage_reason"] == "; ".join(reasons)
        for field in ("eligible_order", "all_20_bijections", "no_fixed_points", "group_membership", "all_metrics_recomputed"):
            assert audit[field] is True
        assert len(audit["donor_map_sha256"]) == 20
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in audit["donor_map_sha256"])
        assert len(set(audit["donor_map_sha256"])) == distinct
        hashes(audit["input_hashes"], "group audit input hashes")
        assert audit["input_hashes"]["summary.json"] == file_hash(path)
        hashes(child["output_sha256"], "child output hashes")
        for field in ("coverage_fraction", "correct_mae", "off_mae", "conditional_mean_mae",
                      "context_global_whole_test_donor_mean_mae", "correct_advantage_pct_vs_conditional",
                      "correct_advantage_pct_vs_global_context", "conditional_worse_than_correct_assignments",
                      "fraction_samples_correct_better_than_conditional_mean", "same_embedding_fraction_mean",
                      "same_description_fraction_mean", "max_rho_final_audited"):
            close(row[field], child[field], f"{key} child {field}")
        correct, off, conditional, global_context = [number(row[field], field) for field in (
            "correct_mae", "off_mae", "conditional_mean_mae", "context_global_whole_test_donor_mean_mae")]
        assert min(correct, off, conditional, global_context) >= 0
        if conditional > 0:
            close(row["correct_advantage_pct_vs_conditional"], 100 * (1 - correct / conditional), f"{key} normalized effect")
        if global_context > 0:
            close(row["correct_advantage_pct_vs_global_context"], 100 * (1 - correct / global_context), f"{key} global normalized effect")
        close(row["conditional_minus_correct_mean_error"], conditional - correct, f"{key} conditional gain")
        close(row["off_minus_correct_mean_error"], off - correct, f"{key} OFF gain")
        close(row["global_context_minus_correct_mean_error"], global_context - correct, f"{key} global gain")
        for field in ("fraction_samples_correct_better_than_conditional_mean", "same_embedding_fraction_mean", "same_description_fraction_mean"):
            assert 0 <= number(row[field], field) <= 1
        assert 0 <= int(row["conditional_worse_than_correct_assignments"]) <= 20
        rho = number(row["max_rho_final_audited"], "group rho")
        assert 0 <= rho <= .1001 and rho <= parent["max_rho_final_audited"] + 1e-8
        assert 0 <= number(child["subset_reproduction_max_abs_pred_diff_native"], "subset replay") <= 1e-5
        for prefix in ("conditional_minus_correct", "off_minus_correct", "global_context_minus_correct", "advantage_pct"):
            low, high = row[prefix + "_ci_low"], row[prefix + "_ci_high"]
            if prefix == "advantage_pct" and low == high == "" and conditional == 0:
                continue
            low, high = number(low, prefix), number(high, prefix)
            assert low <= high, (key, prefix, "inverted CI")
            if prefix == "advantage_pct":
                assert high <= 100 + 1e-10
        perm_path = run / group / "permutation_metrics.csv"
        permutations = rows(perm_path)
        assert len(permutations) == 20 and [int(item["permutation_index"]) for item in permutations] == list(range(20))
        assert file_hash(perm_path) == child["output_sha256"]["permutation_metrics.csv"] == audit["input_hashes"]["permutation_metrics.csv"]
        for name in ("predictions.npz", "sample_metrics.csv"):
            assert child["output_sha256"][name] == audit["input_hashes"][name]
        for item in permutations:
            for name, value in item.items():
                number(value, name)
            assert number(item["mae"], "assignment MAE") >= 0
            close(item["fixed_point_fraction"], 0, f"{key} fixed points")
            for name in ("same_embedding_fraction", "same_description_fraction"):
                assert 0 <= number(item[name], name) <= 1
            for name in ("donor_cosine_mean", "donor_cosine_q05", "donor_cosine_q95"):
                assert -1 - 1e-10 <= number(item[name], name) <= 1 + 1e-10
            assert number(item["donor_cosine_q05"], "cosine quantile") <= number(item["donor_cosine_q95"], "cosine quantile")
        assignment_maes = [number(item["mae"], "assignment MAE") for item in permutations]
        close(statistics.mean(assignment_maes), conditional, f"{key} mean assignment MAE")
        close(statistics.stdev(assignment_maes), child["conditional_mae_sd_over_assignments"], f"{key} assignment sample SD")
        for name in ("same_embedding", "same_description"):
            close(statistics.mean(number(item[name + "_fraction"], name) for item in permutations), row[name + "_fraction_mean"], f"{key} {name} mean")
        close(statistics.mean(number(item["donor_cosine_mean"], "cosine") for item in permutations), child["donor_cosine_mean"], f"{key} donor cosine")

    aggregate = json_file(folder / "aggregate.json")
    assert aggregate["status"] == "PASS" and aggregate["complete"] is True
    assert aggregate["fixed_required_results"] == aggregate["audited_results"] == 28
    assert aggregate["no_causal_composition_percentage_decomposition"] is True
    assert aggregate["no_physical_mae_pooling"] is True
    assert aggregate["bootstrap"]["draws"] == 2000 and aggregate["bootstrap"]["unit"] == "paired crystal"
    assert aggregate["bootstrap"]["assignment_errors_averaged_first"] is True
    assert aggregate["bootstrap"]["composition_cluster_dependence_accounted_for"] is False
    summaries = {(item["group_type"], int(item["seed"])): item for item in aggregate["group_seed_summaries"]}
    assert len(aggregate["group_seed_summaries"]) == len(summaries) == 6
    assert set(summaries) == {(group, seed) for group in group_types for seed in (42, 7, 1234)}
    for (group_type, seed), summary in summaries.items():
        subset = [row for row in data if row["group_type"] == group_type and int(row["seed"]) == seed]
        complete = [row for row in subset if row["status"] == "complete"]
        effects = [number(row["correct_advantage_pct_vs_conditional"], "task effect") for row in complete
                   if row["correct_advantage_pct_vs_conditional"] != ""]
        assert summary["required_tasks"] == len(subset) == summary["reported_tasks"] == (6 if seed == 42 else 4)
        assert summary["eligible_tasks"] == len(complete)
        assert summary["ineligible_tasks"] == sum(row["status"] == "ineligible" for row in subset)
        assert summary["tasks_correct_better_than_conditional"] == sum(value > 0 for value in effects)
        assert summary["tasks_correct_not_better_than_conditional"] == sum(value <= 0 for value in effects)
        assert summary["weak_coverage_tasks"] == sum(boolean(row["weak_coverage"]) for row in complete)
        if effects:
            close(summary["mean_task_advantage_pct"], statistics.mean(effects), f"{group_type}/{seed} normalized macro")
        else:
            assert summary["mean_task_advantage_pct"] is None
    return len(data), len([row for row in data if row["status"] == "complete"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="refresh derived summaries")
    args = parser.parse_args()
    sft = audit_sft(args.write)
    alignn = audit_alignn(args.write)
    pairing = audit_semantic_pairing()
    output = audit_output_correction()
    composition = audit_composition_pairing()
    print(f"SFTGNN macro={sft[0]:.10f} sample_sd={sft[1]:.10f} improved={sft[2]}/30")
    print(f"ALIGNN  macro={alignn[0]:.10f} sample_sd={alignn[1]:.10f} improved={alignn[2]}/30")
    print(f"PAIRING seed42_mean_advantage={pairing[0]:.10f}% positive={pairing[1]}/{pairing[1]}")
    print(f"OUTPUT  audited_runs={output[0]} correct_vs_OFF_benefit={output[1][0]:.6f}..{output[1][1]:.6f}")
    print(f"COMPOSITION audited_groups={composition[0]} eligible_groups={composition[1]}")


if __name__ == "__main__":
    main()
