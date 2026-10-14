"""Sequential, resumable Phase-A screening queue."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp/TextResidualSHFMat-FAIR')
EXP = ROOT / 'experiments/backbone_transfer_alignn'
PHASE = EXP / 'phaseA_screen'
TASKS = [
    'Jarvis-Bandgap_MBJ',
    'Jarvis-FormationEnergy',
    'Jarvis-BulkModulusKv',
    'MP-Bandgap',
    'MP-FormationEnergy',
]
PYTHON = '/root/venvs/fair-v21/bin/python'


def write_status(payload):
    payload['updated_at'] = datetime.now().isoformat(timespec='seconds')
    (EXP / 'queue_status.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main():
    PHASE.mkdir(parents=True, exist_ok=True)
    (EXP / 'logs').mkdir(parents=True, exist_ok=True)
    done=[]
    for task in TASKS:
        missing=[m for m in ('graph','early','hard') if not (PHASE/task/m/'metrics.json').exists()]
        if not missing:
            done.append(task); continue
        write_status({'phase':'A','state':'running','current_task':task,'missing_models':missing,'done_tasks':done})
        log_path=EXP/'logs'/f'phaseA_{task}.log'
        cmd=[PYTHON, str(EXP/'train_alignn_transfer.py'), '--task',task,'--models',*missing,
             '--seed','42','--epochs','60','--batch-size','32','--output-root',str(PHASE)]
        env=dict(os.environ, PYTHONPATH=f"{ROOT/'third_party/alignn'}:{EXP}", CUDA_VISIBLE_DEVICES='0',
                 OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        with log_path.open('a',encoding='utf-8') as log:
            code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if code:
            write_status({'phase':'A','state':'failed','current_task':task,'exit_code':code,'log':str(log_path),'done_tasks':done})
            raise SystemExit(code)
        done.append(task)
    write_status({'phase':'A','state':'training_complete','done_tasks':done})
    code=subprocess.call([PYTHON,str(EXP/'summarize.py'),'--phase','A'],cwd=ROOT,
                         env=dict(os.environ,PYTHONPATH=f"{ROOT/'third_party/alignn'}:{EXP}"))
    if code:
        raise SystemExit(code)


if __name__=='__main__':
    main()
