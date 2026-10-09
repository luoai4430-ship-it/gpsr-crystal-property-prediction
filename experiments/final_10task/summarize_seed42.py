from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "final_10task"
SUMMARY = EXP / "summary"
TASKS = [
    "Jarvis-Bandgap_MBJ",
    "Jarvis-Bandgap_OPT",
    "Jarvis-FormationEnergy",
    "Jarvis-TotalEnergy",
    "Jarvis-BulkModulusKv",
    "Jarvis-ShearModulusGv",
    "MP-Bandgap",
    "MP-FormationEnergy",
    "MP-BulkModuli",
    "MP-ShearModuli",
]
MODELS = ["graph", "textresidual", "early_nodegate", "final_hard_010"]
MODEL_LABELS = {
    "graph": "Graph",
    "textresidual": "TextResidual",
    "early_nodegate": "Early NodeGate",
    "final_hard_010": "Final Hard-0.10",
}
CATEGORIES = {
    "Electronic": ["Jarvis-Bandgap_MBJ", "Jarvis-Bandgap_OPT", "MP-Bandgap"],
    "Energy": ["Jarvis-FormationEnergy", "Jarvis-TotalEnergy", "MP-FormationEnergy"],
    "Mechanical": [
        "Jarvis-BulkModulusKv",
        "Jarvis-ShearModulusGv",
        "MP-BulkModuli",
        "MP-ShearModuli",
    ],
}
OLD6 = [
    "Jarvis-Bandgap_MBJ",
    "Jarvis-FormationEnergy",
    "Jarvis-BulkModulusKv",
    "MP-Bandgap",
    "MP-FormationEnergy",
    "MP-ShearModuli",
]
NEW4 = ["Jarvis-Bandgap_OPT", "Jarvis-TotalEnergy", "Jarvis-ShearModulusGv", "MP-BulkModuli"]


def run_metrics(model: str, task: str) -> dict:
    directory = EXP / "seed42_benchmark" / model / task
    train_files = glob.glob(str(directory / "*_train_metrics.csv"))
    test_files = glob.glob(str(directory / "*_test_metrics.csv"))
    if len(train_files) != 1 or len(test_files) != 1:
        raise RuntimeError(f"Expected one train/test metrics file in {directory}")
    train = pd.read_csv(train_files[0])
    best_index = train.valid_mae.astype(float).idxmin()
    best = train.loc[best_index]
    test = pd.read_csv(test_files[0]).iloc[0]
    return {
        "best_epoch": int(best.epoch),
        "valid_mae": float(best.valid_mae),
        "test_mae": float(test.MAE),
        "train_metrics_file": train_files[0],
        "test_metrics_file": test_files[0],
    }


def write_markdown_table(frame: pd.DataFrame, path: Path, title: str) -> None:
    path.write_text(f"# {title}\n\n" + frame.to_markdown(index=False) + "\n", encoding="utf-8")


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    records = []
    raw = {(task, model): run_metrics(model, task) for task in TASKS for model in MODELS}
    for task in TASKS:
        graph_test = raw[(task, "graph")]["test_mae"]
        text_test = raw[(task, "textresidual")]["test_mae"]
        for model in MODELS:
            metric = raw[(task, model)]
            records.append(
                {
                    "task": task,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "seed": 42,
                    "best_epoch": metric["best_epoch"],
                    "valid_mae": metric["valid_mae"],
                    "test_mae": metric["test_mae"],
                    "normalized_mae": metric["test_mae"] / graph_test,
                    "gain_vs_graph_pct": (graph_test - metric["test_mae"]) / graph_test * 100,
                    "gain_vs_textresidual_pct": (text_test - metric["test_mae"]) / text_test * 100,
                }
            )
    detail = pd.DataFrame(records)
    detail.to_csv(SUMMARY / "benchmark_seed42.csv", index=False)

    valid_table = pd.DataFrame({"Task": TASKS})
    test_table = pd.DataFrame({"Task": TASKS})
    for model in MODELS:
        valid_table[MODEL_LABELS[model]] = [raw[(task, model)]["valid_mae"] for task in TASKS]
        test_table[MODEL_LABELS[model]] = [raw[(task, model)]["test_mae"] for task in TASKS]
    test_table["Final gain vs Graph %"] = [
        (raw[(task, "graph")]["test_mae"] - raw[(task, "final_hard_010")]["test_mae"])
        / raw[(task, "graph")]["test_mae"]
        * 100
        for task in TASKS
    ]
    test_table["Final gain vs TextResidual %"] = [
        (raw[(task, "textresidual")]["test_mae"] - raw[(task, "final_hard_010")]["test_mae"])
        / raw[(task, "textresidual")]["test_mae"]
        * 100
        for task in TASKS
    ]
    valid_table.to_csv(SUMMARY / "benchmark_seed42_valid_wide.csv", index=False)
    test_table.to_csv(SUMMARY / "benchmark_seed42_test_wide.csv", index=False)
    write_markdown_table(valid_table, SUMMARY / "benchmark_seed42_valid.md", "10-task Seed-42 Validation MAE")
    write_markdown_table(test_table, SUMMARY / "benchmark_seed42_test.md", "10-task Seed-42 Test MAE")

    macro_rows = []
    category_rows = []
    for model in MODELS:
        ratios = {
            task: raw[(task, model)]["test_mae"] / raw[(task, "graph")]["test_mae"]
            for task in TASKS
        }
        macro = float(np.mean(list(ratios.values())))
        improved = sum(raw[(task, model)]["test_mae"] < raw[(task, "graph")]["test_mae"] for task in TASKS)
        macro_rows.append(
            {
                "model": MODEL_LABELS[model],
                "macro_normalized_mae": macro,
                "avg_gain_pct": (1 - macro) * 100,
                "improved_vs_graph": improved,
            }
        )
        category_rows.append(
            {
                "model": MODEL_LABELS[model],
                **{
                    category: float(np.mean([ratios[task] for task in tasks]))
                    for category, tasks in CATEGORIES.items()
                },
                "Overall": macro,
                "Old_6": float(np.mean([ratios[task] for task in OLD6])),
                "New_4": float(np.mean([ratios[task] for task in NEW4])),
            }
        )
    macro_frame = pd.DataFrame(macro_rows)
    category_frame = pd.DataFrame(category_rows)
    macro_frame.to_csv(SUMMARY / "benchmark_macro.csv", index=False)
    category_frame.to_csv(SUMMARY / "benchmark_category_macro.csv", index=False)

    diagnostics = []
    violations = []
    for task in TASKS:
        path = EXP / "seed42_benchmark" / "final_hard_010" / task / "gpsr_test_stats.json"
        stats = json.loads(path.read_text(encoding="utf-8"))
        row = {"task": task, **stats}
        diagnostics.append(row)
        if float(stats.get("max_rho_final", float("inf"))) > 0.1001:
            violations.append({"task": task, "max_rho_final": stats.get("max_rho_final")})
    pd.DataFrame(diagnostics).to_csv(SUMMARY / "final_hard_diagnostics.csv", index=False)
    if violations:
        raise RuntimeError(f"Hard residual bound violation: {violations}")

    shuffled_rows = []
    for task in ("MP-Bandgap", "MP-FormationEnergy"):
        files = glob.glob(str(EXP / "shuffled_controls" / task / "*_test_metrics.csv"))
        if len(files) != 1:
            raise RuntimeError(f"Missing shuffled test metrics for {task}")
        shuffled_mae = float(pd.read_csv(files[0]).iloc[0].MAE)
        correct_mae = raw[(task, "final_hard_010")]["test_mae"]
        shuffled_rows.append(
            {
                "task": task,
                "correct_text_mae": correct_mae,
                "shuffled_text_mae": shuffled_mae,
                "correct_gain_pct": (shuffled_mae - correct_mae) / shuffled_mae * 100,
                "correct_better": correct_mae < shuffled_mae,
            }
        )
    pd.DataFrame(shuffled_rows).to_csv(SUMMARY / "mp_correct_vs_shuffled.csv", index=False)

    final_macro = next(row["macro_normalized_mae"] for row in macro_rows if row["model"] == MODEL_LABELS["final_hard_010"])
    final_improved = next(row["improved_vs_graph"] for row in macro_rows if row["model"] == MODEL_LABELS["final_hard_010"])
    launch_multiseed = final_improved >= 7 and final_macro < 0.98
    status = {
        "stage_b_complete": True,
        "final_improved_tasks": int(final_improved),
        "final_macro_normalized_mae": float(final_macro),
        "success_threshold_met": bool(launch_multiseed),
        "launch_multiseed": bool(launch_multiseed),
        "hard_bound_all_tasks_pass": not violations,
        "correct_vs_shuffled": shuffled_rows,
    }
    (EXP / "stage_b_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    report = [
        "# Final 10-task benchmark: Seed 42",
        "",
        f"Final Hard-0.10 improves {final_improved}/10 tasks versus Graph, with Macro NMAE {final_macro:.6f}.",
        f"Stage-C trigger: {'PASS' if launch_multiseed else 'NOT MET'} (requires >=7/10 and Macro NMAE < 0.98).",
        "",
        "## Macro",
        "",
        macro_frame.to_markdown(index=False),
        "",
        "## Category macro",
        "",
        category_frame.to_markdown(index=False),
        "",
        "## Correct vs shuffled on repaired MP text",
        "",
        pd.DataFrame(shuffled_rows).to_markdown(index=False),
        "",
    ]
    (SUMMARY / "seed42_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
