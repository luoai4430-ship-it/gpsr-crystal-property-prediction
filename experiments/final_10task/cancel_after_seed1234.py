#!/usr/bin/env python3
import json, os, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/final_10task"
STATUS = EXP / "queue_status.json"
LOG = EXP / "logs" / "cancel_2025_2026.log"

def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")

def process_table():
    out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,cmd="], text=True)
    rows=[]
    for line in out.splitlines():
        parts=line.strip().split(None,2)
        if len(parts)==3:
            rows.append((int(parts[0]),int(parts[1]),parts[2]))
    return rows

def descendants(root, rows):
    children={}
    for pid,ppid,cmd in rows: children.setdefault(ppid,[]).append(pid)
    found=[]; stack=list(children.get(root,[]))
    while stack:
        p=stack.pop(); found.append(p); stack.extend(children.get(p,[]))
    return found

log("watchdog armed; waiting for seed 1234 to finish")
while True:
    try:
        q=json.loads(STATUS.read_text(encoding="utf-8"))
        phase=str(q.get("phase",""))
    except Exception:
        time.sleep(0.05); continue
    if phase in {"multiseed_seed_2025", "multiseed_seed_2026"}:
        rows=process_table()
        managers=[pid for pid,ppid,cmd in rows if cmd.endswith("experiments/final_10task/multiseed_queue.py") or "/experiments/final_10task/multiseed_queue.py" in cmd]
        log(f"detected prohibited phase {phase}; managers={managers}")
        for manager in managers:
            ds=descendants(manager,rows)
            for pid in reversed(ds):
                try: os.kill(pid,signal.SIGTERM)
                except ProcessLookupError: pass
            try: os.kill(manager,signal.SIGTERM)
            except ProcessLookupError: pass
        time.sleep(2)
        subprocess.run(["tmux","kill-session","-t","final10_pipeline"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        # Remove only canceled, incomplete run directories; completed 42/7/1234 artifacts remain untouched.
        for seed in (2025,2026):
            p=EXP/"multiseed"/f"seed_{seed}"
            if p.exists():
                import shutil; shutil.rmtree(p)
        (EXP/"user_scope.json").write_text(json.dumps({"seeds":[42,7,1234],"canceled_seeds":[2025,2026],"requested_at":datetime.now().isoformat()},indent=2),encoding="utf-8")
        command = f"exec {os.environ.get('GPSR_PYTHON', sys.executable)} experiments/final_10task/pipeline.py >> experiments/final_10task/logs/pipeline_stdout.log 2>&1"
        subprocess.run(["tmux","new-session","-d","-s","final10_pipeline", command], cwd=ROOT, check=True)
        log("restarted pipeline with seeds 42,7,1234 only")
        break
    time.sleep(0.05)
