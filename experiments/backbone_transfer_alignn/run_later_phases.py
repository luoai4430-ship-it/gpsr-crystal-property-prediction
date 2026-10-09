"""Gate and run Phase B and Phase C without changing frozen settings."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/backbone_transfer_alignn'
PYTHON=os.environ.get('GPSR_PYTHON',sys.executable)
TASKS=[
 'Jarvis-Bandgap_MBJ','Jarvis-Bandgap_OPT','Jarvis-FormationEnergy','Jarvis-TotalEnergy',
 'Jarvis-BulkModulusKv','Jarvis-ShearModulusGv','MP-Bandgap','MP-FormationEnergy',
 'MP-BulkModuli','MP-ShearModuli']
PHASE_A={'Jarvis-Bandgap_MBJ','Jarvis-FormationEnergy','Jarvis-BulkModulusKv','MP-Bandgap','MP-FormationEnergy'}


def env():
    return dict(os.environ,PYTHONPATH=f'{ROOT}/third_party/alignn:{EXP}',CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')


def status(**kwargs):
    kwargs['updated_at']=datetime.now().isoformat(timespec='seconds')
    (EXP/'queue_status.json').write_text(json.dumps(kwargs,indent=2),encoding='utf-8')


def run(cmd,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as handle:
        return subprocess.call(cmd,cwd=ROOT,env=env(),stdout=handle,stderr=subprocess.STDOUT)


def link_phase_a_into_b():
    root=EXP/'phaseB_10task'
    for task in PHASE_A:
        for mode in ('graph','hard'):
            destination=root/task/mode
            source=EXP/'phaseA_screen'/task/mode
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists() and source.exists():
                destination.symlink_to(source,target_is_directory=True)


def link_seed42_into_c():
    root=EXP/'phaseC_multiseed'/'seed_42'
    for task in TASKS:
        for mode in ('graph','hard'):
            destination=root/task/mode
            source=EXP/'phaseB_10task'/task/mode
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists() and source.exists():
                destination.symlink_to(source,target_is_directory=True)


def main():
    a=json.loads((EXP/'phaseA_status.json').read_text())
    if a['verdict']!='BACKBONE_TRANSFER_GAIN':
        subprocess.check_call([PYTHON,str(EXP/'summarize_full.py'),'--stop-after','A'],cwd=ROOT,env=env())
        status(state='complete',stopped_after='A',verdict=a['verdict'])
        return

    link_phase_a_into_b()
    done=[]
    for task in TASKS:
        missing=[m for m in ('graph','hard') if not (EXP/'phaseB_10task'/task/m/'metrics.json').exists()]
        if not missing:
            done.append(task);continue
        status(phase='B',state='running',current_task=task,missing_models=missing,done_tasks=done)
        cmd=[PYTHON,str(EXP/'train_alignn_transfer.py'),'--task',task,'--models',*missing,'--seed','42',
             '--epochs','60','--batch-size','32','--output-root',str(EXP/'phaseB_10task')]
        code=run(cmd,EXP/'logs'/f'phaseB_{task}.log')
        if code:
            status(phase='B',state='failed',current_task=task,exit_code=code);raise SystemExit(code)
        done.append(task)
    subprocess.check_call([PYTHON,str(EXP/'summarize_full.py'),'--stop-after','B'],cwd=ROOT,env=env())
    b=json.loads((EXP/'phaseB_status.json').read_text())
    if not b['phase_c_triggered']:
        status(state='complete',stopped_after='B',verdict=b['verdict'])
        return

    link_seed42_into_c()
    done=[]
    for task in TASKS:
        missing=[]
        for seed in (7,1234):
            for mode in ('graph','hard'):
                if not (EXP/'phaseC_multiseed'/f'seed_{seed}'/task/mode/'metrics.json').exists():
                    missing.append((seed,mode))
        if not missing:
            done.append(task);continue
        status(phase='C',state='running',current_task=task,missing=[f'{s}/{m}' for s,m in missing],done_tasks=done)
        seeds=sorted({s for s,_ in missing})
        # A task is built once and all missing model/seed combinations are run in one process.
        # Normally both models are missing for both seeds in this fixed matrix.
        models=sorted({m for _,m in missing})
        cmd=[PYTHON,str(EXP/'train_multiseed_task.py'),'--task',task,'--seeds',*map(str,seeds),
             '--models',*models,'--epochs','60','--batch-size','32','--output-root',str(EXP/'phaseC_multiseed')]
        code=run(cmd,EXP/'logs'/f'phaseC_{task}.log')
        if code:
            status(phase='C',state='failed',current_task=task,exit_code=code);raise SystemExit(code)
        done.append(task)
    subprocess.check_call([PYTHON,str(EXP/'summarize_full.py'),'--stop-after','C'],cwd=ROOT,env=env())
    final=json.loads((EXP/'completion_status.json').read_text())
    status(state='complete',phase='C',verdict=final['verdict'])


if __name__=='__main__':
    main()
