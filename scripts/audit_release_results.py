#!/usr/bin/env python3
"""Audit and optionally refresh compact GPSR release summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments/final_10task/summary/multiseed_paired_results.csv"
ALIGNN = ROOT / "experiments/backbone_transfer_alignn/summary/alignn_multiseed_paired.csv"
PAIRING = ROOT / "experiments/semantic_pairing_permutation/task_summary.csv"


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
    assert len(pairs) == 14
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="refresh derived summaries")
    args = parser.parse_args()
    sft = audit_sft(args.write)
    alignn = audit_alignn(args.write)
    pairing = audit_semantic_pairing()
    print(f"SFTGNN macro={sft[0]:.10f} sample_sd={sft[1]:.10f} improved={sft[2]}/30")
    print(f"ALIGNN  macro={alignn[0]:.10f} sample_sd={alignn[1]:.10f} improved={alignn[2]}/30")
    print(f"PAIRING seed42_mean_advantage={pairing[0]:.10f}% positive={pairing[1]}/{pairing[1]}")


if __name__ == "__main__":
    main()
