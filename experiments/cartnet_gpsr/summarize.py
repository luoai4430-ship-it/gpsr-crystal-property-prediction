"""Create phase gates, required CSVs, and the honest strong-backbone report."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/cartnet_gpsr'
from data_cartnet import TASKS

def readj(path): return json.loads(path.read_text())
def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('');return
    fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def phase0():
    rows=[]
    for p in sorted((EXP/'phase0_official_reproduction').glob('*/metrics.json')):
        rows.append(readj(p))
    write_csv(EXP/'summary/cartnet_reproduction_check.csv',rows)
    passed=len(rows)==4 and all(r['sanity_pass'] for r in rows)
    result={'phase':'phase0','complete':len(rows)==4,'sanity_pass':passed,
            'tasks_completed':len(rows),'failed_tasks':[r['task'] for r in rows if not r['sanity_pass']]}
    (EXP/'phase0_status.json').write_text(json.dumps(result,indent=2))
    return result

def phase1():
    rows=[]; diagnostics=[]
    for task in TASKS:
        pair={}
        for mode in ('graph','gpsr'):
            p=EXP/'phase1_matched_10task'/task/mode/'metrics.json'
            if p.exists(): pair[mode]=readj(p)
        if len(pair)!=2: continue
        g=pair['graph']['test_mae']; s=pair['gpsr']['test_mae']; ratio=s/g
        rows.append({'task':task,'category':TASKS[task]['category'],'seed':pair['graph']['seed'],
                     'cartnet_mae':g,'cartnet_gpsr_mae':s,'gnmae':ratio,
                     'relative_gain_percent':100*(1-ratio),'improved':s<g})
        d=EXP/'phase1_matched_10task'/task/'gpsr'/'diagnostics.json'
        if d.exists(): diagnostics.append({'phase':'phase1','task':task,'seed':pair['gpsr']['seed'],**readj(d)})
    write_csv(EXP/'summary/cartnet_seed1_10task.csv',rows)
    write_csv(EXP/'diagnostics/cartnet_gpsr_diagnostics.csv',diagnostics)
    if len(rows)!=10:
        result={'phase':'phase1','complete':False,'tasks_completed':len(rows),'proceed_phase2':False}
    else:
        macro=float(np.mean([r['gnmae'] for r in rows])); improved=sum(r['improved'] for r in rows)
        maxrho=max((d.get('max_rho_final',math.inf) for d in diagnostics),default=math.inf)
        if macro<=.98 and improved>=7 and maxrho<=.1001: verdict='STRONG_BACKBONE_GAIN'; proceed=True
        elif macro<1 and improved>=6 and maxrho<=.1001: verdict='MODERATE_STRONG_BACKBONE_GAIN'; proceed=True
        elif macro<1: verdict='PARTIAL_TRANSFER'; proceed=False
        else: verdict='NO_STRONG_BACKBONE_GAIN'; proceed=False
        result={'phase':'phase1','complete':True,'macro_gnmae':macro,'mean_gain_percent':100*(1-macro),
                'improved_tasks':improved,'max_rho_final':maxrho,'verdict':verdict,'proceed_phase2':proceed}
    (EXP/'phase1_status.json').write_text(json.dumps(result,indent=2))
    return result

def phase2():
    paired=[]; diag=[]
    for seed in (42,7,1234):
        for task in TASKS:
            base=EXP/'phase1_matched_10task'/task if seed==42 else EXP/'phase2_multirun'/f'seed{seed}'/task
            ps={m:base/m/'metrics.json' for m in ('graph','gpsr')}
            if not all(p.exists() for p in ps.values()): continue
            a={m:readj(p) for m,p in ps.items()};g=a['graph']['test_mae'];s=a['gpsr']['test_mae']
            paired.append({'task':task,'category':TASKS[task]['category'],'seed':seed,'cartnet_mae':g,
                           'cartnet_gpsr_mae':s,'gnmae':s/g,'relative_gain_percent':100*(1-s/g),'improved':s<g})
            dp=base/'gpsr'/'diagnostics.json'
            if dp.exists():diag.append({'phase':'phase2','task':task,'seed':seed,**readj(dp)})
    task_rows=[]
    for task in TASKS:
        rr=[r for r in paired if r['task']==task]
        if not rr:continue
        task_rows.append({'task':task,'category':TASKS[task]['category'],'runs':len(rr),
                          'cartnet_mean':np.mean([r['cartnet_mae'] for r in rr]),'cartnet_std':np.std([r['cartnet_mae'] for r in rr],ddof=1) if len(rr)>1 else 0,
                          'gpsr_mean':np.mean([r['cartnet_gpsr_mae'] for r in rr]),'gpsr_std':np.std([r['cartnet_gpsr_mae'] for r in rr],ddof=1) if len(rr)>1 else 0,
                          'mean_paired_gain_percent':np.mean([r['relative_gain_percent'] for r in rr]),'improved_runs':sum(r['improved'] for r in rr)})
    macro=[]
    for seed in (42,7,1234):
        rr=[r for r in paired if r['seed']==seed]
        if len(rr)==10:macro.append({'seed':seed,'macro_gnmae':np.mean([r['gnmae'] for r in rr]),'improved':sum(r['improved'] for r in rr)})
    macro_summary=[]
    if macro:
        vals=[r['macro_gnmae'] for r in macro]
        macro_summary=[{'runs':len(vals),'macro_gnmae_mean':np.mean(vals),'macro_gnmae_std':np.std(vals,ddof=1) if len(vals)>1 else 0,
                        'mean_gain_percent':100*(1-np.mean(vals)),'lower_mae_evaluations':sum(r['improved'] for r in paired),'total_evaluations':len(paired)}]
    write_csv(EXP/'summary/cartnet_multirun_task_summary.csv',task_rows)
    write_csv(EXP/'summary/cartnet_multirun_macro_summary.csv',macro_summary)
    write_csv(EXP/'summary/cartnet_multirun_paired.csv',paired)
    old=[]
    p=EXP/'diagnostics/cartnet_gpsr_diagnostics.csv'
    if p.exists() and p.stat().st_size:
        old=list(csv.DictReader(p.open()))
    write_csv(p,old+diag)
    return macro_summary[0] if macro_summary else {}

def final_report():
    p0=phase0();p1=phase1();p2=phase2()
    cart_macro=p2.get('macro_gnmae_mean',p1.get('macro_gnmae',float('nan')))
    cart_gain=100*(1-cart_macro) if math.isfinite(cart_macro) else float('nan')
    cart_imp=f"{p2.get('lower_mae_evaluations',p1.get('improved_tasks',0))}/{p2.get('total_evaluations',10 if p1.get('complete') else 0)}"
    three=[{'backbone':'SFTGNN','macro_gnmae':.9664,'mean_gain_percent':3.36,'improved_evaluations':'28/30'},
           {'backbone':'ALIGNN','macro_gnmae':.9703,'mean_gain_percent':2.97,'improved_evaluations':'26/30'},
           {'backbone':'CartNet','macro_gnmae':cart_macro,'mean_gain_percent':cart_gain,'improved_evaluations':cart_imp}]
    write_csv(EXP/'summary/three_backbone_summary.csv',three)
    report=f"""# CartNet + GPSR strong-backbone validation

## Protocol

Official CartNet revision and software provenance are recorded in `preflight.json`. The official scalar configuration is used: cutoff 5.0, all neighbors within the radius, four interaction layers, hidden width 256, Adam, OneCycleLR, 500 epochs, batch size 64, and minimum-validation-MAE checkpoint selection. GPSR is injected only after `CartNet.layers[0]`, with frozen 768-dimensional MatBERT inputs, NodeGate, channel-wise alpha initialized to 1e-3, and sample-wise hard projection at tau=0.10.

## Phase 0

Completed anchors: {p0.get('tasks_completed',0)}/4. Sanity verdict: **{'SANITY_PASS' if p0.get('sanity_pass') else 'SANITY_FAIL or incomplete'}**.

## Matched fixed-split result

Phase-1 verdict: **{p1.get('verdict','incomplete')}**. Macro GN-MAE: {p1.get('macro_gnmae',float('nan')):.4f}; mean relative gain: {p1.get('mean_gain_percent',float('nan')):.2f}%; improved tasks: {p1.get('improved_tasks',0)}/10; maximum observed rho_final: {p1.get('max_rho_final',float('nan')):.6f}.

## Three-run result

Macro GN-MAE mean: {p2.get('macro_gnmae_mean',float('nan')):.4f} ± {p2.get('macro_gnmae_std',float('nan')):.4f}. Lower-MAE matched evaluations: {p2.get('lower_mae_evaluations',0)}/{p2.get('total_evaluations',0)}.

## Interpretation rule

Claims must follow the preregistered gate. A positive result supports effectiveness on a recent high-performing backbone. It does not establish universal backbone agnosticism or state of the art. Published CartNet values are literature-scale references unless dataset version, split, and target transformation are identical.
"""
    (EXP/'summary/cartnet_gpsr_report.md').write_text(report,encoding='utf-8')
    return {'phase0':p0,'phase1':p1,'phase2':p2}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=('phase0','phase1','final'),default='final');x=a.parse_args()
    print(json.dumps(phase0() if x.stage=='phase0' else phase1() if x.stage=='phase1' else final_report(),indent=2))
