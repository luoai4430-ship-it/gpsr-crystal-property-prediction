from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "final_10task"
PYTHON = os.environ.get("GPSR_PYTHON", sys.executable)


def run(script: Path, log_name: str) -> None:
    log_path = EXP / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {script}\n")
        handle.flush()
        subprocess.run(
            [PYTHON, str(script)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
        )
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] COMPLETE {script}\n")


def main() -> None:
    audit_path = EXP / "text_audit" / "completion_status.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("all_pass"):
        raise RuntimeError("Stage A is not 10/10 PASS")
    status_path = EXP / "completion_status.json"
    status_path.write_text(
        json.dumps({"status": "RUNNING", "stage": "preflight", "updated_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )
    run(EXP / "text_audit" / "preflight_shared_graph_tasks.py", "preflight.log")
    status_path.write_text(
        json.dumps({"status": "RUNNING", "stage": "seed42", "updated_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )
    run(EXP / "seed42_queue.py", "seed42_pipeline.log")
    run(EXP / "summarize_seed42.py", "seed42_summary.log")
    stage_b = json.loads((EXP / "stage_b_status.json").read_text(encoding="utf-8"))
    status_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "stage": "multiseed" if stage_b.get("launch_multiseed") else "final_summary",
                "stage_b": stage_b,
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    run(EXP / "multiseed_queue.py", "multiseed_pipeline.log")
    run(EXP / "summarize_final.py", "final_summary.log")
    verdict = json.loads((EXP / "summary" / "verdict.json").read_text(encoding="utf-8"))
    status_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "stage": "complete",
                "completed_at": datetime.now().isoformat(),
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
