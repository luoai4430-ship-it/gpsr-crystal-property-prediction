"""Run the fixed 10-task seed-42 benchmark and two repaired-MP shuffled controls."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
from datetime import datetime


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP = os.path.join(ROOT, "experiments", "final_10task")
PYTHON = "/root/venvs/fair-v21/bin/python"
DATA = "/root/autodl-tmp/dataset1/dataset"
TASKS = {
    "Jarvis-Bandgap_MBJ": "jarvis/mbj_bandgap",
    "Jarvis-Bandgap_OPT": "jarvis/optb88vdw_bandgap",
    "Jarvis-FormationEnergy": "jarvis/formation_energy_peratom",
    "Jarvis-TotalEnergy": "jarvis/optb88vdw_total_energy",
    "Jarvis-BulkModulusKv": "jarvis/bulk_modulus_kv",
    "Jarvis-ShearModulusGv": "jarvis/shear_modulus_gv",
    "MP-Bandgap": "mp_2018/MP-bandgap",
    "MP-FormationEnergy": "mp_2018/MP-formation",
    "MP-BulkModuli": "mp_2018_small/bulk",
    "MP-ShearModuli": "mp_2018_small/shear",
}
MODELS = {
    "graph": ("graphonly", "SFTGNN"),
    "textresidual": ("textresidual", "TextResidualSHFMat"),
    "early_nodegate": ("gpsrnodegate-early", "GPSRNodeGateEarly"),
    "final_hard_010": ("gpsrnodegate-early-bounded", "GPSRNodeGateEarlyBoundedTau010"),
}


def complete(out: str) -> bool:
    return bool(glob.glob(os.path.join(out, "*_test_metrics.csv"))) and os.path.isfile(
        os.path.join(out, "best_valid.pth")
    )


def collect(out: str, queue_log_path: str, task: str, prefix: str, started_at: float) -> None:
    run_dir = None
    with open(queue_log_path, errors="ignore") as handle:
        for line in handle:
            if "Run output directory:" in line:
                run_dir = line.split("Run output directory:", 1)[1].strip()
    if run_dir and not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT, run_dir)
    if run_dir and os.path.isdir(run_dir):
        for name in os.listdir(run_dir):
            source = os.path.join(run_dir, name)
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(out, "run_detail.log" if name == "run.log" else name))
        shutil.rmtree(run_dir, ignore_errors=True)

    parameter_prefix = os.path.join(ROOT, "parameter", f"{prefix}_{task}")
    best_path = parameter_prefix + "_para.pth"
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"best-valid checkpoint missing: {best_path}")
    shutil.copy2(best_path, os.path.join(out, "best_valid.pth"))
    candidates = [
        path for path in glob.glob(parameter_prefix + "_top1_*_para.pth")
        if os.path.getmtime(path) >= started_at - 5
    ]
    if candidates:
        shutil.copy2(max(candidates, key=os.path.getmtime), os.path.join(out, "top1.pth"))
    for path in glob.glob(parameter_prefix + "_epoch*_para.pth"):
        if os.path.getmtime(path) >= started_at - 5:
            os.remove(path)
    for path in [best_path, *candidates]:
        if os.path.isfile(path) and os.path.getmtime(path) >= started_at - 5:
            os.remove(path)


def command(task: str, rel: str, model_key: str, seed: int, shuffled: bool = False) -> list[str]:
    cli_model, _ = MODELS[model_key]
    cmd = [
        PYTHON,
        "run.py",
        "--task",
        "train_and_test",
        "--target",
        task,
        "--model",
        cli_model,
        "--use_cif_csv",
        "--cif_dir",
        os.path.join(DATA, rel, "cif"),
        "--text_emb_path",
        os.path.join(DATA, rel, "MatBert_embeddings.pt"),
        "--num_epoch",
        "60",
        "--batch_size",
        "32",
        "--seed",
        str(seed),
        "--lr",
        "0.001",
        "--max_lr",
        "0.001",
        "--weight_decay",
        "0.0001",
    ]
    if model_key == "graph":
        cmd.append("--share_text_cache")
    if model_key in ("early_nodegate", "final_hard_010"):
        cmd += ["--inject_position", "early"]
    if model_key == "final_hard_010":
        cmd += ["--max_residual_ratio", "0.10"]
    if shuffled:
        cmd += ["--text_shuffle_seed", "42"]
    return cmd


def run_phase(
    jobs: list[dict], phase: str, concurrency: int = 3, stage: str = "seed42_benchmark"
) -> list[dict]:
    os.makedirs(os.path.join(EXP, "logs"), exist_ok=True)
    status_path = os.path.join(EXP, "queue_status.json")
    master_log = open(os.path.join(EXP, "logs", "queue.log"), "a", encoding="utf-8")
    pending = [job for job in jobs if not complete(job["out"])]
    running = {}
    done = []
    failed = []
    launched_total = len(pending)

    def emit(message: str) -> None:
        line = f"[{datetime.now():%F %T}] [{phase}] {message}"
        print(line, flush=True)
        master_log.write(line + "\n")
        master_log.flush()

    def status() -> None:
        with open(status_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stage": stage,
                    "phase": phase,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "total_jobs_this_phase": launched_total,
                    "done": done,
                    "failed": failed,
                    "running": [{"key": key, "pid": item["proc"].pid} for key, item in running.items()],
                    "pending": [item["key"] for item in pending],
                },
                handle,
                indent=2,
            )

    emit(f"launching {launched_total} missing jobs; concurrency={concurrency}")
    while pending or running:
        while pending and len(running) < concurrency:
            job = pending.pop(0)
            os.makedirs(job["out"], exist_ok=True)
            log_path = os.path.join(job["out"], "run.log")
            log_handle = open(log_path, "w")
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="0",
                OMP_NUM_THREADS="3",
                MKL_NUM_THREADS="3",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            )
            started_at = time.time()
            proc = subprocess.Popen(
                job["cmd"], cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env=env
            )
            running[job["key"]] = {
                **job,
                "proc": proc,
                "log_handle": log_handle,
                "log_path": log_path,
                "started_at": started_at,
            }
            emit(f"start {job['key']} pid={proc.pid}")
            status()
        time.sleep(10)
        for key, item in list(running.items()):
            code = item["proc"].poll()
            if code is None:
                continue
            item["log_handle"].close()
            if code == 0:
                try:
                    collect(
                        item["out"], item["log_path"], item["task"], item["prefix"], item["started_at"]
                    )
                    done.append(key)
                    emit(f"done {key}")
                except Exception as error:
                    failed.append({"key": key, "exit_code": 0, "collection_error": repr(error)})
                    emit(f"COLLECT_FAIL {key}: {error!r}")
            else:
                failed.append({"key": key, "exit_code": code, "log": item["log_path"]})
                emit(f"FAIL {key} exit={code}")
            del running[key]
            status()
    emit(f"phase finished done={len(done)} failed={len(failed)}")
    master_log.close()
    status()
    return failed


def make_main_jobs() -> list[dict]:
    jobs = []
    for task, rel in TASKS.items():
        for model_key, (_, prefix) in MODELS.items():
            out = os.path.join(EXP, "seed42_benchmark", model_key, task)
            jobs.append(
                {
                    "key": f"seed42/{model_key}/{task}",
                    "task": task,
                    "model": model_key,
                    "prefix": prefix,
                    "out": out,
                    "cmd": command(task, rel, model_key, 42),
                }
            )
    return jobs


def make_control_jobs() -> list[dict]:
    jobs = []
    for task in ("MP-Bandgap", "MP-FormationEnergy"):
        rel = TASKS[task]
        _, prefix = MODELS["final_hard_010"]
        out = os.path.join(EXP, "shuffled_controls", task)
        jobs.append(
            {
                "key": f"shuffled/{task}",
                "task": task,
                "model": "final_hard_010",
                "prefix": prefix,
                "out": out,
                "cmd": command(task, rel, "final_hard_010", 42, shuffled=True),
            }
        )
    return jobs


def main() -> None:
    audit_status = os.path.join(EXP, "text_audit", "completion_status.json")
    if not os.path.isfile(audit_status) or not json.load(open(audit_status)).get("all_pass"):
        raise RuntimeError("Stage A is not 10/10 PASS; refusing to start Stage B")
    failed = run_phase(make_main_jobs(), "main", concurrency=3)
    if failed:
        raise SystemExit(2)
    failed = run_phase(make_control_jobs(), "shuffled_controls", concurrency=2)
    if failed:
        raise SystemExit(3)
    with open(os.path.join(EXP, "seed42_queue_complete.json"), "w", encoding="utf-8") as handle:
        json.dump({"complete": True, "main_jobs": 40, "shuffled_jobs": 2}, handle, indent=2)


if __name__ == "__main__":
    main()
