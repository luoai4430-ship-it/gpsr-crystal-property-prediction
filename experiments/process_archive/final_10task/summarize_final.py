from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "final_10task"
SUMMARY = EXP / "summary"
PLOTS = EXP / "plots"
TASKS = [
    "Jarvis-Bandgap_MBJ", "Jarvis-Bandgap_OPT", "Jarvis-FormationEnergy",
    "Jarvis-TotalEnergy", "Jarvis-BulkModulusKv", "Jarvis-ShearModulusGv",
    "MP-Bandgap", "MP-FormationEnergy", "MP-BulkModuli", "MP-ShearModuli",
]
SEEDS = [42, 7, 1234]


def test_mae(seed: int, model: str, task: str) -> float:
    if seed == 42:
        directory = EXP / "seed42_benchmark" / model / task
    else:
        directory = EXP / "multiseed" / f"seed_{seed}" / model / task
    files = glob.glob(str(directory / "*_test_metrics.csv"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one test metrics file in {directory}")
    return float(pd.read_csv(files[0]).iloc[0].MAE)


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    stage_b = json.loads((EXP / "stage_b_status.json").read_text(encoding="utf-8"))
    audit = json.loads((EXP / "text_audit" / "completion_status.json").read_text(encoding="utf-8"))
    multiseed_status_path = EXP / "multiseed" / "completion_status.json"
    multiseed_status = json.loads(multiseed_status_path.read_text()) if multiseed_status_path.exists() else {}
    seeds = multiseed_status.get("seeds", SEEDS) if multiseed_status.get("complete") else [42]

    paired_rows = []
    task_rows = []
    macro_rows = []
    for seed in seeds:
        ratios = []
        for task in TASKS:
            graph = test_mae(seed, "graph", task)
            final = test_mae(seed, "final_hard_010", task)
            ratio = final / graph
            ratios.append(ratio)
            paired_rows.append(
                {
                    "seed": seed,
                    "task": task,
                    "graph_mae": graph,
                    "final_mae": final,
                    "paired_ratio_final_over_graph": ratio,
                    "relative_gain_pct": (1 - ratio) * 100,
                    "improved": final < graph,
                }
            )
        macro_rows.append(
            {
                "seed": seed,
                "macro_normalized_mae": float(np.mean(ratios)),
                "macro_gain_pct": float((1 - np.mean(ratios)) * 100),
                "improved_tasks": int(sum(ratio < 1 for ratio in ratios)),
            }
        )
    paired = pd.DataFrame(paired_rows)
    for task in TASKS:
        subset = paired[paired.task == task]
        task_rows.append(
            {
                "task": task,
                "graph_mean_mae": subset.graph_mae.mean(),
                "graph_std_mae": subset.graph_mae.std(ddof=0),
                "final_mean_mae": subset.final_mae.mean(),
                "final_std_mae": subset.final_mae.std(ddof=0),
                "mean_paired_ratio": subset.paired_ratio_final_over_graph.mean(),
                "std_paired_ratio": subset.paired_ratio_final_over_graph.std(ddof=0),
                "relative_gain_pct": (1 - subset.paired_ratio_final_over_graph.mean()) * 100,
                "improved_seeds_count": int(subset.improved.sum()),
                "num_seeds": len(subset),
            }
        )
    task_summary = pd.DataFrame(task_rows)
    macro_by_seed = pd.DataFrame(macro_rows)
    macro_summary = pd.DataFrame(
        [
            {
                "num_seeds": len(seeds),
                "macro_normalized_mae_mean": macro_by_seed.macro_normalized_mae.mean(),
                "macro_normalized_mae_std": macro_by_seed.macro_normalized_mae.std(ddof=0),
                "macro_gain_pct_mean": macro_by_seed.macro_gain_pct.mean(),
                "mean_paired_improvement": float((1 - paired.paired_ratio_final_over_graph).mean()),
                "median_paired_improvement": float((1 - paired.paired_ratio_final_over_graph).median()),
            }
        ]
    )
    try:
        from scipy.stats import wilcoxon

        test = wilcoxon(paired.paired_ratio_final_over_graph - 1.0, alternative="less")
        macro_summary["wilcoxon_statistic"] = float(test.statistic)
        macro_summary["wilcoxon_pvalue_one_sided"] = float(test.pvalue)
    except Exception as error:
        macro_summary["wilcoxon_error"] = repr(error)
    paired.to_csv(SUMMARY / "multiseed_paired_results.csv", index=False)
    task_summary.to_csv(SUMMARY / "multiseed_task_summary.csv", index=False)
    macro_by_seed.to_csv(SUMMARY / "multiseed_macro_by_seed.csv", index=False)
    macro_summary.to_csv(SUMMARY / "multiseed_macro_summary.csv", index=False)

    seed42 = pd.read_csv(SUMMARY / "benchmark_seed42.csv")
    main_rows = []
    for task in TASKS:
        row = {"dataset_property": task}
        for model, label in (
            ("graph", "Graph"),
            ("textresidual", "TextResidual"),
            ("early_nodegate", "Early NodeGate"),
            ("final_hard_010", "Final Hard-0.10"),
        ):
            value = float(seed42[(seed42.task == task) & (seed42.model == model)].test_mae.iloc[0])
            if model in ("graph", "final_hard_010") and len(seeds) > 1:
                summary_row = task_summary[task_summary.task == task].iloc[0]
                mean = summary_row.graph_mean_mae if model == "graph" else summary_row.final_mean_mae
                std = summary_row.graph_std_mae if model == "graph" else summary_row.final_std_mae
                row[label] = f"{mean:.6g} ± {std:.3g}"
            else:
                row[label] = f"{value:.6g} (seed 42)"
        main_rows.append(row)
    pd.DataFrame(main_rows).to_csv(SUMMARY / "paper_main_table.csv", index=False)

    gains42 = paired[paired.seed == 42]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(TASKS)), gains42.relative_gain_pct)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(TASKS)), TASKS, rotation=35, ha="right")
    ax.set_ylabel("Final gain vs Graph (%)")
    ax.set_title("Final Hard-0.10: Seed-42 gain across 10 tasks")
    fig.tight_layout()
    fig.savefig(PLOTS / "seed42_task_gain.png", dpi=220)
    plt.close(fig)
    if len(seeds) > 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(np.arange(len(TASKS)), task_summary.relative_gain_pct)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(TASKS)), TASKS, rotation=35, ha="right")
        ax.set_ylabel("Mean paired gain (%)")
        ax.set_title(f"Final Hard-0.10: {len(seeds)}-seed paired gain")
        fig.tight_layout()
        fig.savefig(PLOTS / "multiseed_task_gain.png", dpi=220)
        plt.close(fig)

    controls = pd.read_csv(SUMMARY / "mp_correct_vs_shuffled.csv")
    majority_threshold = len(seeds) // 2 + 1
    majority_stable = int((task_summary.improved_seeds_count >= majority_threshold).sum())
    correspondence = bool(controls.correct_better.all())
    seed42_improved = int(stage_b["final_improved_tasks"])
    seed42_macro = float(stage_b["final_macro_normalized_mae"])
    multi_macro = float(macro_summary.macro_normalized_mae_mean.iloc[0])
    if (
        len(seeds) >= 3
        and seed42_improved >= 8
        and seed42_macro <= 0.97
        and majority_stable >= 6
        and correspondence
    ):
        verdict = "STRONG_FINAL_MODEL"
    elif seed42_improved >= 6 and seed42_macro < 1.0 and multi_macro < 1.0:
        verdict = "FINAL_MODEL_GAIN"
    elif seed42_improved > 0 or multi_macro < 1.0:
        verdict = "PARTIAL_FINAL_GAIN"
    else:
        verdict = "NO_FINAL_GAIN"
    verdict_obj = {
        "verdict": verdict,
        "text_audit_passed": bool(audit.get("all_pass")),
        "seed42_improved_tasks": seed42_improved,
        "seed42_macro_normalized_mae": seed42_macro,
        "multiseed_completed": bool(multiseed_status.get("complete")),
        "multiseed_num_seeds": len(seeds),
        "multiseed_macro_mean": multi_macro,
        "multiseed_macro_std": float(macro_summary.macro_normalized_mae_std.iloc[0]),
        "tasks_with_majority_seeds_improved": majority_stable if len(seeds) > 1 else None,
        "majority_seed_threshold": majority_threshold if len(seeds) > 1 else None,
        "canceled_seeds": [2025, 2026],
        "mp_correct_text_better_than_shuffled_both": correspondence,
        "hard_bound_all_tasks_pass": bool(stage_b.get("hard_bound_all_tasks_pass")),
    }
    (SUMMARY / "verdict.json").write_text(json.dumps(verdict_obj, indent=2), encoding="utf-8")
    category = pd.read_csv(SUMMARY / "benchmark_category_macro.csv")
    report = [
        "# Final 10-task experiment report",
        "",
        f"**Verdict: {verdict}**",
        "",
        f"Text audit: {audit.get('passed')}/{audit.get('total')} PASS.",
        f"Seed 42: Final Hard-0.10 improves {seed42_improved}/10 tasks; Macro NMAE = {seed42_macro:.6f}.",
        f"Multi-seed: {'completed' if multiseed_status.get('complete') else 'not completed'} ({len(seeds)} seeds); Macro NMAE = {multi_macro:.6f}.",
        "Seeds 2025 and 2026 were canceled by user request; robustness scope is seeds 42, 7, and 1234.",
        "",
        "## Seed-42 category macro",
        "",
        category.to_markdown(index=False),
        "",
        "## Robustness by task",
        "",
        task_summary.to_markdown(index=False),
        "",
        "## Repaired MP text correspondence",
        "",
        controls.to_markdown(index=False),
        "",
        "## Hard residual control",
        "",
        "All ten tasks satisfy max rho_final <= 0.1001." if stage_b.get("hard_bound_all_tasks_pass") else "A hard-bound violation was detected.",
        "",
    ]
    (SUMMARY / "final_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(verdict_obj, indent=2))


if __name__ == "__main__":
    main()
