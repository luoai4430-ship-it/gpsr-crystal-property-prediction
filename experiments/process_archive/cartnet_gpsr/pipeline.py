"""Strict four-phase CartNet validation pipeline with preregistered gates."""
from __future__ import annotations
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; EXP=ROOT/'experiments/cartnet_gpsr'; PY='/root/venvs/fair-v21/bin/python'
TASKS=['Jarvis-Bandgap_MBJ','Jarvis-Bandgap_OPT','Jarvis-FormationEnergy','Jarvis-TotalEnergy',
       'MP-Bandgap','MP-FormationEnergy','MP-BulkModuli','MP-ShearModuli','Jarvis-BulkModulusKv','Jarvis-ShearModulusGv']
PHASE0=['JARVIS-MBJ','JARVIS-OPT','MP-Bandgap','MP-Bulk']

def complete(out):return (out/'metrics.json').is_file() and (out/'best_validation.pt').is_file()
def status(stage,current=None,done=None,pending=None,failed=None):
    obj={'stage':stage,'updated_at':datetime.now().isoformat(timespec='seconds'),'current':current,
         'done':done or [],'pending':pending or [],'failed':failed}
    (EXP/'queue_status.json').write_text(json.dumps(obj,indent=2))
def run(key,cmd,out):
    out.mkdir(parents=True,exist_ok=True)
    log=out/'run.log'; status('running',key)
    with log.open('a',encoding='utf-8') as h:
        h.write(f"\n[{datetime.now().isoformat()}] COMMAND {' '.join(map(str,cmd))}\n");h.flush()
        p=subprocess.run(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
    if p.returncode: raise RuntimeError(f'{key} failed exit={p.returncode}; log={log}')
def call_summary(stage):
    return subprocess.run([PY,str(EXP/'summarize.py'),'--stage',stage],cwd=ROOT,check=True)

def main():
    (EXP/'logs').mkdir(parents=True,exist_ok=True)
    subprocess.run([PY,str(EXP/'preflight.py')],cwd=ROOT,check=True,
                   stdout=(EXP/'logs/preflight.log').open('w'),stderr=subprocess.STDOUT)
    done=[]
    try:
        for task in PHASE0:
            out=EXP/'phase0_official_reproduction'/task
            if not complete(out):run(f'phase0/{task}',[PY,str(EXP/'train_phase0.py'),'--task',task,'--out',str(out)],out)
            done.append(f'phase0/{task}')
        call_summary('phase0');p0=json.loads((EXP/'phase0_status.json').read_text())
        if not p0['sanity_pass']:
            status('stopped_sanity_fail',done=done,failed=p0);return
        for task in TASKS:
            for mode in ('graph','gpsr'):
                out=EXP/'phase1_matched_10task'/task/mode
                key=f'phase1/{task}/{mode}'
                if not complete(out):run(key,[PY,str(EXP/'train_matched.py'),'--task',task,'--mode',mode,'--seed','42','--out',str(out)],out)
                done.append(key)
        call_summary('phase1');p1=json.loads((EXP/'phase1_status.json').read_text())
        if not p1.get('proceed_phase2'):
            call_summary('final');status('complete_gate_stop',done=done,failed=None);return
        for seed in (7,1234):
            for task in TASKS:
                for mode in ('graph','gpsr'):
                    out=EXP/'phase2_multirun'/f'seed{seed}'/task/mode;key=f'phase2/seed{seed}/{task}/{mode}'
                    if not complete(out):run(key,[PY,str(EXP/'train_matched.py'),'--task',task,'--mode',mode,'--seed',str(seed),'--out',str(out)],out)
                    done.append(key)
        call_summary('final');status('complete',done=done)
        (EXP/'completion_status.json').write_text(json.dumps({'complete':True,'completed_at':datetime.now().isoformat()},indent=2))
    except Exception as e:
        status('failed',done=done,failed=repr(e));raise
if __name__=='__main__':main()
