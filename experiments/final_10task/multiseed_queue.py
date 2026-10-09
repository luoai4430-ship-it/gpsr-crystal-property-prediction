"""Run Graph and fixed Final Hard-0.10 on the user-selected non-42 seeds."""
from __future__ import annotations

import json
import os

from seed42_queue import EXP, MODELS, TASKS, command, run_phase


SEEDS = [7, 1234]


def jobs_for_seed(seed: int) -> list[dict]:
    jobs = []
    for task, rel in TASKS.items():
        for model_key in ("graph", "final_hard_010"):
            _, prefix = MODELS[model_key]
            out = os.path.join(EXP, "multiseed", f"seed_{seed}", model_key, task)
            jobs.append(
                {
                    "key": f"seed{seed}/{model_key}/{task}",
                    "task": task,
                    "model": model_key,
                    "prefix": prefix,
                    "out": out,
                    "cmd": command(task, rel, model_key, seed),
                }
            )
    return jobs


def main() -> None:
    status_path = os.path.join(EXP, "stage_b_status.json")
    if not os.path.isfile(status_path):
        raise RuntimeError("Stage B summary is missing")
    stage_b = json.load(open(status_path))
    if not stage_b.get("launch_multiseed"):
        with open(os.path.join(EXP, "multiseed", "completion_status.json"), "w") as handle:
            json.dump({"complete": False, "skipped": True, "reason": "Stage-B trigger not met"}, handle, indent=2)
        return
    for seed in SEEDS:
        failed = run_phase(
            jobs_for_seed(seed),
            f"multiseed_seed_{seed}",
            concurrency=3,
            stage="multiseed",
        )
        if failed:
            raise SystemExit(10 + SEEDS.index(seed))
    output = os.path.join(EXP, "multiseed", "completion_status.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            {"complete": True, "seeds": [42, *SEEDS], "models": ["graph", "final_hard_010"], "runs": 20 * len([42, *SEEDS]), "scope_note": "Seeds 2025 and 2026 canceled by user"},
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
