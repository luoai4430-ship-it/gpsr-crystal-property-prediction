"""Generate gated Phase-B/C summaries and the final portability report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/backbone_transfer_alignn'; SUMMARY=EXP/'summary'
TASKS=[
 'Jarvis-Bandgap_MBJ','Jarvis-Bandgap_OPT','Jarvis-FormationEnergy','Jarvis-TotalEnergy',
 'Jarvis-BulkModulusKv','Jarvis-ShearModulusGv','MP-Bandgap','MP-FormationEnergy',
 'MP-BulkModuli','MP-ShearModuli']
CATEGORY={
 'Jarvis-Bandgap_MBJ':'Electronic','Jarvis-Bandgap_OPT':'Electronic','MP-Bandgap':'Electronic',
 'Jarvis-FormationEnergy':'Energy','Jarvis-TotalEnergy':'Energy','MP-FormationEnergy':'Energy',
 'Jarvis-BulkModulusKv':'Mechanical','Jarvis-ShearModulusGv':'Mechanical',
 'MP-BulkModuli':'Mechanical','MP-ShearModuli':'Mechanical'}


def metric(root,task,mode):
    return json.loads((root/task/mode/'metrics.json').read_text())


def phase_b():
    rows=[];diag=[]
    root=EXP/'phaseB_10task'
    for task in TASKS:
        g=metric(root,task,'graph');h=metric(root,task,'hard')
        rows.append({'task':task,'category':CATEGORY[task],'seed':42,'graph_valid_mae':g['valid_mae'],
          'hard_valid_mae':h['valid_mae'],'graph_test_mae':g['test_mae'],'hard_test_mae':h['test_mae'],
          'hard_nmae':h['test_mae']/g['test_mae'],'gain_pct':(1-h['test_mae']/g['test_mae'])*100,
          'improved':h['test_mae']<g['test_mae']})
        diag.append({'phase':'B','seed':42,**h})
    df=pd.DataFrame(rows);df.to_csv(SUMMARY/'alignn_10task_seed42.csv',index=False)
    macro=float(df.hard_nmae.mean());improved=int(df.improved.sum())
    cats=df.groupby('category').hard_nmae.mean().to_dict()
    bound=all(float(x['max_rho_final'])<=0.1001 for x in diag)
    triggered=macro<0.985 and improved>=7 and bound
    verdict='BACKBONE_TRANSFER_GAIN_10TASK' if triggered else ('PARTIAL_BACKBONE_GENERALIZATION' if macro<1 else 'BACKBONE_SPECIFIC_GAIN')
    result={'phase':'B','verdict':verdict,'macro_nmae':macro,'gain_pct':(1-macro)*100,
      'improved_tasks':improved,'num_tasks':10,'electronic_nmae':cats.get('Electronic'),
      'energy_nmae':cats.get('Energy'),'mechanical_nmae':cats.get('Mechanical'),
      'hard_bound_pass':bound,'phase_c_triggered':triggered}
    (EXP/'phaseB_status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame([
      {'backbone':'SHFMat','tasks':10,'graph_nmae':1.0,'hard_nmae':0.9568534715163338,'gain_pct':4.31465284836662,'improved_tasks':'10/10'},
      {'backbone':'ALIGNN','tasks':10,'graph_nmae':1.0,'hard_nmae':macro,'gain_pct':(1-macro)*100,'improved_tasks':f'{improved}/10'}
    ]).to_csv(SUMMARY/'backbone_portability_summary.csv',index=False)
    return result,diag


def phase_c():
    paired=[];diag=[]
    root=EXP/'phaseC_multiseed'
    for seed in (42,7,1234):
        for task in TASKS:
            g=metric(root/f'seed_{seed}',task,'graph');h=metric(root/f'seed_{seed}',task,'hard')
            paired.append({'seed':seed,'task':task,'graph_mae':g['test_mae'],'hard_mae':h['test_mae'],
                           'ratio':h['test_mae']/g['test_mae'],'gain_pct':(1-h['test_mae']/g['test_mae'])*100})
            diag.append({'phase':'C','seed':seed,**h})
    p=pd.DataFrame(paired);p.to_csv(SUMMARY/'alignn_multiseed_paired.csv',index=False)
    rows=[]
    for task,x in p.groupby('task',sort=False):
        rows.append({'task':task,'graph_mean_mae':x.graph_mae.mean(),'graph_std_mae':x.graph_mae.std(ddof=1),
          'hard_mean_mae':x.hard_mae.mean(),'hard_std_mae':x.hard_mae.std(ddof=1),
          'mean_paired_ratio':x.ratio.mean(),'mean_gain_pct':x.gain_pct.mean(),
          'improved_seeds_count':int((x.gain_pct>0).sum()),'num_seeds':3})
    summary=pd.DataFrame(rows);summary.to_csv(SUMMARY/'alignn_multiseed_summary.csv',index=False)
    macro_by_seed=p.groupby('seed').ratio.mean()
    macro_mean=float(macro_by_seed.mean());macro_std=float(macro_by_seed.std(ddof=1))
    majority=int((summary.improved_seeds_count>=2).sum())
    bound=all(float(x['max_rho_final'])<=0.1001 for x in diag)
    if macro_mean<1 and majority>=6 and bound:
        verdict='BACKBONE_AGNOSTIC_GRAPH_PRESERVING_GAIN'
    elif macro_mean<1 and majority>=3 and bound:
        verdict='PARTIAL_BACKBONE_GENERALIZATION'
    else:
        verdict='BACKBONE_SPECIFIC_GAIN'
    result={'phase':'C','verdict':verdict,'num_seeds':3,'macro_nmae_mean':macro_mean,
      'macro_nmae_std':macro_std,'gain_pct_mean':(1-macro_mean)*100,
      'tasks_with_majority_seeds_improved':majority,'hard_bound_pass':bound}
    (EXP/'completion_status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result,diag


def report(stop_after,b=None,c=None):
    a=json.loads((EXP/'phaseA_status.json').read_text())
    lines=['# ALIGNN backbone transfer report','',f"Phase A verdict: **{a['verdict']}**",'',
      f"Phase-A Graph/Early/Hard Macro NMAE: 1.000000 / {a['early_macro_nmae']:.6f} / {a['hard_macro_nmae']:.6f}.",
      f"Phase-A Hard improved {a['hard_improved_tasks']}/5 tasks; hard bound pass={a['hard_bound_pass']}.",'']
    if (SUMMARY/'alignn_phaseA_screen.csv').exists():
        lines += ['## Phase A',pd.read_csv(SUMMARY/'alignn_phaseA_screen.csv').to_markdown(index=False),'']
    if b:
        lines += ['## Phase B',f"10-task Macro NMAE={b['macro_nmae']:.6f}; gain={b['gain_pct']:.2f}%; improved={b['improved_tasks']}/10.",
          f"Electronic/Energy/Mechanical NMAE={b['electronic_nmae']:.6f}/{b['energy_nmae']:.6f}/{b['mechanical_nmae']:.6f}.",'',
          pd.read_csv(SUMMARY/'alignn_10task_seed42.csv').to_markdown(index=False),'']
    if c:
        lines += ['## Phase C',f"3-seed Macro NMAE={c['macro_nmae_mean']:.6f} ± {c['macro_nmae_std']:.6f}; gain={c['gain_pct_mean']:.2f}%.",
          f"Tasks improved in at least 2/3 seeds: {c['tasks_with_majority_seeds_improved']}/10.",'',
          pd.read_csv(SUMMARY/'alignn_multiseed_summary.csv').to_markdown(index=False),'',f"**Final verdict: {c['verdict']}**"]
    elif b:
        lines += [f"**Current verdict: {b['verdict']}**"]
    else:
        lines += [f"**Final verdict: {a['verdict']}**"]
    (SUMMARY/'alignn_transfer_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stop-after',choices=['A','B','C'],required=True);args=parser.parse_args()
    SUMMARY.mkdir(parents=True,exist_ok=True)
    b=c=None;diag=[]
    if args.stop_after in ('B','C'):
        b,d=phase_b();diag+=d
    if args.stop_after=='C':
        c,d=phase_c();diag+=d
    # Preserve Phase-A diagnostics and add later hard runs without duplicate task/seed rows.
    phase_a_path=SUMMARY/'alignn_residual_diagnostics.csv'
    frames=[]
    if phase_a_path.exists(): frames.append(pd.read_csv(phase_a_path))
    if diag: frames.append(pd.DataFrame(diag))
    if frames:
        all_diag=pd.concat(frames,ignore_index=True,sort=False)
        keys=[k for k in ('phase','seed','task') if k in all_diag.columns]
        if keys: all_diag=all_diag.drop_duplicates(keys,keep='last')
        all_diag.to_csv(phase_a_path,index=False)
    report(args.stop_after,b,c)


if __name__=='__main__': main()
