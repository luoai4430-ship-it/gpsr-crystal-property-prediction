"""Wait for Phase A, then apply the predeclared phase gates."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/TextResidualSHFMat-FAIR')
EXP=ROOT/'experiments/backbone_transfer_alignn'
while not (EXP/'phaseA_status.json').exists():
    q=EXP/'queue_status.json'
    if q.exists() and json.loads(q.read_text()).get('state')=='failed':
        raise SystemExit(2)
    time.sleep(60)
env=dict(os.environ,PYTHONPATH=f'{ROOT}/third_party/alignn:{EXP}',CUDA_VISIBLE_DEVICES='0',
         OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
raise SystemExit(subprocess.call(
    ['/root/venvs/fair-v21/bin/python',str(EXP/'run_later_phases.py')],cwd=ROOT,env=env
))
