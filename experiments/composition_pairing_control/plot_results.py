"""Static scientific figure for audited frozen-checkpoint pairing controls.

Only final task_summary.csv data are visualized. No model inference, target
selection, or independent-model performance comparison is performed here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TASKS = (
    "Jarvis-Bandgap_MBJ", "Jarvis-Bandgap_OPT", "Jarvis-FormationEnergy",
    "Jarvis-BulkModulusKv", "MP-Bandgap", "MP-BulkModuli",
)
REPEATED = {"Jarvis-Bandgap_MBJ", "Jarvis-FormationEnergy", "Jarvis-BulkModulusKv", "MP-Bandgap"}
GROUPS = ("reduced_composition", "element_set")
SHORT = {
    "Jarvis-Bandgap_MBJ": "J-MBJ",
    "Jarvis-Bandgap_OPT": "J-OPT",
    "Jarvis-FormationEnergy": "J-Formation",
    "Jarvis-BulkModulusKv": "J-Bulk",
    "MP-Bandgap": "MP-Bandgap",
    "MP-BulkModuli": "MP-Bulk",
}
COLORS = {42: "#0072B2", 7: "#D55E00", 1234: "#009E73"}
ORDER = tuple((task, seed) for task in TASKS for seed in ((42, 7, 1234) if task in REPEATED else (42,)))
EXPECTED = {(task, seed, group) for task, seed in ORDER for group in GROUPS}
CAPTION = (
    "Frozen, validation-selected SFTGNN-GPSR checkpoints; each comparison uses its group-eligible test subset. "
    "Correct advantage = 100 × (1 − MAE_correct / mean MAE_conditional); positive values favor correct pairing. "
    "Error bars are descriptive 95% paired crystal-bootstrap intervals (2,000 resamples; conditional absolute errors "
    "averaged within each crystal). Composition-cluster and donor dependence remains. Twenty draws may repeat donor "
    "maps. These exploratory within-checkpoint controls do not compare independently trained models. "
    "J: JARVIS; MP: Materials Project; MBJ/OPT: band-gap settings; Formation: formation energy; Bulk: bulk modulus."
)


def number(row, field):
    value = row.get(field, "")
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {field}: {row['task']} seed={row['seed']}")
    return result


def load_rows(root: Path):
    path = root / "task_summary.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {}
    for row in rows:
        key = (row["task"], int(row["seed"]), row["group_type"])
        if key in by_key:
            raise ValueError(f"Duplicate final summary key: {key}")
        if row["status"] not in ("complete", "ineligible"):
            raise ValueError(f"Final summary has incomplete status: {key}")
        by_key[key] = row
    if set(by_key) != EXPECTED or len(rows) != 28:
        raise ValueError(f"Expected exactly the fixed 28 final results; missing={sorted(EXPECTED - set(by_key))}, "
                         f"extra={sorted(set(by_key) - EXPECTED)}")
    audit_file = root / "validation.json"
    if audit_file.is_file():
        audit = json.loads(audit_file.read_text(encoding="utf-8-sig"))
        if audit.get("status") != "PASS" or audit.get("complete") is not True:
            raise ValueError("Final result audit is not complete/PASS")
    return by_key


def draw(root: Path):
    rows = load_rows(root)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(16, 9.7), sharex=True)
    fig.subplots_adjust(left=.19, right=.97, top=.80, bottom=.23, wspace=.78)
    x_values = [0.]
    for row in rows.values():
        if row["status"] == "complete":
            for field in ("correct_advantage_pct_vs_conditional", "advantage_pct_ci_low", "advantage_pct_ci_high"):
                value = number(row, field)
                if value is not None:
                    x_values.append(value)
    left, right = min(x_values), max(x_values)
    span = max(right - left, 10.)
    limits = (left - .10 * span, right + .10 * span)
    titles = ("(a) Exact reduced composition (primary)", "(b) Element set (secondary)")
    for ax, group_type, title in zip(axes, GROUPS, titles):
        labels = []
        for y, (task, seed) in enumerate(ORDER):
            row = rows[(task, seed, group_type)]
            n = int(row["num_eligible"])
            distinct = int(row["num_distinct_assignments"])
            labels.append(f"{SHORT[task]} · seed {seed}  (N={n})")
            if y and task != ORDER[y - 1][0]:
                ax.axhline(y - .5, color="#dddddd", linewidth=.8, zorder=0)
            if row["status"] == "ineligible":
                ax.text(.98, y, "Ineligible: no donor peer", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=9, color="#777777")
                continue
            effect = number(row, "correct_advantage_pct_vs_conditional")
            low, high = number(row, "advantage_pct_ci_low"), number(row, "advantage_pct_ci_high")
            if effect is None:
                ax.text(.98, y, "Undefined: conditional MAE = 0", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=9, color="#777777")
                continue
            if low is not None and high is not None:
                if low > high:
                    raise ValueError(f"Reversed confidence interval: {task}, seed={seed}, {group_type}")
                # Draw interval directly so percentile CIs need not contain
                # their point estimate; no truncation or negative-xerr repair.
                ax.hlines(y, low, high, color=COLORS[seed], linewidth=1.5, alpha=.80, zorder=2)
                ax.vlines((low, high), y - .075, y + .075, color=COLORS[seed], linewidth=1.2, zorder=2)
            weak = n < 30 or distinct == 1
            ax.plot(effect, y, marker="o", markersize=7.5, markeredgewidth=1.6,
                    markeredgecolor=COLORS[seed], markerfacecolor="white" if weak else COLORS[seed],
                    linestyle="none", zorder=3)
        ax.set_yticks(range(len(ORDER)), labels)
        ax.set_ylim(len(ORDER) - .45, -.55)
        ax.set_xlim(*limits)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.1, zorder=1)
        ax.grid(axis="x", color="#eeeeee", linewidth=.8, zorder=0)
        ax.set_title(title, pad=12)
        ax.set_xlabel("Correct-pairing MAE advantage over conditional pairing (%)", labelpad=11)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.spines["left"].set_visible(False)
    fig.suptitle("Composition-conditioned semantic pairing", fontsize=18, y=.972)
    fig.text(.5, .935, "Frozen-checkpoint control on composition-eligible test subsets", ha="center", fontsize=12)
    legend = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                     markerfacecolor=COLORS[seed], markeredgecolor=COLORS[seed], label=f"Training seed {seed}")
              for seed in (42, 7, 1234)]
    legend.append(Line2D([], [], marker="o", linestyle="none", markersize=7,
                         markerfacecolor="white", markeredgecolor="#333333", markeredgewidth=1.4,
                         label="Open marker: eligible N < 30 or only one distinct donor map"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .90), ncol=4,
               frameon=False, fontsize=10, columnspacing=1.8, handletextpad=.6)
    caption = textwrap.fill(CAPTION, width=164)
    fig.text(.07, .125, caption, ha="left", va="top", fontsize=10, linespacing=1.45)
    png, pdf = root / "mechanism_control.png", root / "mechanism_control.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=.2)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.2,
                metadata={"Title": "Composition-conditioned semantic pairing control",
                          "Subject": "Exploratory frozen-checkpoint subset control; descriptive crystal-bootstrap intervals"})
    plt.close(fig)
    print(json.dumps({"figure_png": str(png), "figure_pdf": str(pdf), "results": 28}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    draw(args.root.resolve())


if __name__ == "__main__":
    main()
