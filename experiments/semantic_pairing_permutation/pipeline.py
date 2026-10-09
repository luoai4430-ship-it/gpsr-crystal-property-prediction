"""Run seed-42 semantic-pairing controls, then optional matched multi-seed follow-up."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/semantic_pairing_permutation"
SOURCE = ROOT / "experiments/final_confirmatory/semantic_intervention/runs"
DATA = Path("/root/autodl-tmp/dataset1/dataset")
PYTHON = "/root/venvs/fair-v21/bin/python"
TASKS = {
    "Jarvis-Bandgap_MBJ": "jarvis/mbj_bandgap",
    "Jarvis-Bandgap_OPT": "jarvis/optb88vdw_bandgap",
    "Jarvis-FormationEnergy": "jarvis/formation_energy_peratom",
    "Jarvis-BulkModulusKv": "jarvis/bulk_modulus_kv",
    "MP-Bandgap": "mp_2018/MP-bandgap",
    "MP-BulkModuli": "mp_2018_small/bulk",
}
FOLLOWUP = ["Jarvis-Bandgap_MBJ", "Jarvis-FormationEnergy", "Jarvis-BulkModulusKv", "MP-Bandgap"]


def stamp(message: str):
    line = f"[{datetime.now():%F %T}] {message}"
    print(line, flush=True)
    with (EXP / "pipeline.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(stage, **kwargs):
    (EXP / "queue_status.json").write_text(json.dumps({
        "stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds"), **kwargs
    }, indent=2), encoding="utf-8")


def run_one(task: str, seed: int, permutations: int):
    rel = TASKS[task]
    source = SOURCE / f"seed_{seed}/gpsr/{task}"
    out = EXP / "runs" / f"seed_{seed}" / task
    if (out / "summary.json").exists():
        stamp(f"reuse seed={seed} task={task}")
        return
    checkpoint = source / "best_valid.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(EXP / "evaluate.py"), "--target", task,
           "--checkpoint", str(checkpoint), "--run-dir", str(source),
           "--cif-dir", str(DATA / rel / "cif"),
           "--text-emb-path", str(DATA / rel / "MatBert_embeddings.pt"),
           "--seed", str(seed), "--permutations", str(permutations), "--out", str(out)]
    log = (out / "run.log").open("w", encoding="utf-8")
    stamp(f"start seed={seed} task={task} permutations={permutations}")
    code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    log.close()
    if code:
        raise RuntimeError(f"failed seed={seed} task={task} exit={code}")
    stamp(f"done seed={seed} task={task}")


def aggregate():
    rows = [json.loads(p.read_text()) for p in sorted((EXP / "runs").glob("seed_*/*/summary.json"))]
    df = pd.DataFrame(rows).sort_values(["seed", "task"])
    df.to_csv(EXP / "task_summary.csv", index=False)
    seed42 = df[df.seed.eq(42)]
    aggregate = {
        "num_runs": int(len(df)), "num_seed42_tasks": int(len(seed42)),
        "seed42_mean_pairing_advantage_pct": float(seed42.pairing_advantage_pct_vs_shuffle.mean()),
        "seed42_tasks_positive_pairing_advantage": int((seed42.pairing_advantage_pct_vs_shuffle > 0).sum()),
        "seed42_tasks_permutation_p_le_0_05": int((seed42.permutation_p_one_sided <= .05).sum()),
        "max_formal_reproduction_abs_diff_standardized": float(df.formal_reproduction_max_abs_pred_diff_standardized.max()),
    }
    (EXP / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    lines = ["# Semantic Pairing Permutation Control", "",
             "Frozen validation-selected GPSR checkpoints were evaluated with correctly paired text, semantic OFF, and within-test-set derangements of MatBERT embeddings. No model was trained or selected in this analysis.", "",
             df[["task", "seed", "correct_mae", "off_mae", "shuffle_mean_mae", "pairing_advantage_pct_vs_shuffle", "permutation_p_one_sided", "fraction_samples_pairing_helped"]].to_markdown(index=False), "",
             f"Seed-42 pairing advantage was positive on {aggregate['seed42_tasks_positive_pairing_advantage']}/{aggregate['num_seed42_tasks']} tasks; mean advantage {aggregate['seed42_mean_pairing_advantage_pct']:.3f}%.", "",
             "Interpretation is exploratory and within-checkpoint. A positive pairing advantage indicates that preserving the crystal-text correspondence outperformed distribution-preserving shuffled text under the same frozen parameters; it does not establish a causal benefit relative to an independently trained graph-only model."]
    (EXP / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--permutations", type=int, default=20)
    p.add_argument("--followup", action="store_true")
    a = p.parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    status("seed42", pending=list(TASKS))
    for i, task in enumerate(TASKS):
        status("seed42", running=task, completed=i, total=len(TASKS))
        run_one(task, 42, a.permutations)
    if a.followup:
        for seed in (7, 1234):
            for task in FOLLOWUP:
                status("followup", running=f"{seed}/{task}")
                run_one(task, seed, a.permutations)
    agg = aggregate()
    status("complete", aggregate=agg)
    stamp("pipeline complete")


if __name__ == "__main__":
    main()
