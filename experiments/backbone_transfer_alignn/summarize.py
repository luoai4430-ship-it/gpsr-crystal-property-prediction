"""Summarize phase gates and emit the required screening files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/backbone_transfer_alignn'


def phase_a():
    rows=[]; diagnostics=[]
    phase=EXP/'phaseA_screen'
    tasks=['Jarvis-Bandgap_MBJ','Jarvis-FormationEnergy','Jarvis-BulkModulusKv','MP-Bandgap','MP-FormationEnergy']
    for task in tasks:
        metrics={}
        for mode in ('graph','early','hard'):
            p=phase/task/mode/'metrics.json'
            if not p.exists():
                raise FileNotFoundError(p)
            metrics[mode]=json.loads(p.read_text())
            if mode=='hard': diagnostics.append(metrics[mode])
        graph=metrics['graph']['test_mae']; early=metrics['early']['test_mae']; hard=metrics['hard']['test_mae']
        rows.append({
            'task':task,
            'graph_valid_mae':metrics['graph']['valid_mae'], 'early_valid_mae':metrics['early']['valid_mae'], 'hard_valid_mae':metrics['hard']['valid_mae'],
            'graph_test_mae':graph,'early_test_mae':early,'hard_test_mae':hard,
            'early_gain_vs_graph_pct':(1-early/graph)*100,
            'hard_gain_vs_graph_pct':(1-hard/graph)*100,
            'hard_gain_vs_early_pct':(1-hard/early)*100,
            'early_nmae':early/graph,'hard_nmae':hard/graph,
        })
    df=pd.DataFrame(rows)
    (EXP/'summary').mkdir(parents=True,exist_ok=True)
    df.to_csv(EXP/'summary/alignn_phaseA_screen.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(EXP/'summary/alignn_residual_diagnostics.csv',index=False)
    hard_macro=float(df.hard_nmae.mean()); early_macro=float(df.early_nmae.mean())
    improved=int((df.hard_test_mae<df.graph_test_mae).sum())
    bound=all(float(x['max_rho_final'])<=0.1001 for x in diagnostics)
    if hard_macro<0.99 and improved>=4 and bound:
        verdict='BACKBONE_TRANSFER_GAIN'
    elif hard_macro<1.0 and improved==3 and bound:
        verdict='PARTIAL_BACKBONE_TRANSFER'
    elif early_macro<1.0 and hard_macro>=1.0:
        verdict='BOUND_NOT_TRANSFERRED'
    else:
        verdict='NO_BACKBONE_TRANSFER'
    summary={'phase':'A','verdict':verdict,'graph_macro_nmae':1.0,'early_macro_nmae':early_macro,
             'hard_macro_nmae':hard_macro,'hard_improved_tasks':improved,'num_tasks':5,'hard_bound_pass':bound}
    (EXP/'phaseA_status.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (EXP/'queue_status.json').write_text(json.dumps({'phase':'A','state':'complete',**summary},indent=2),encoding='utf-8')
    report=['# ALIGNN backbone transfer – Phase A','',f"**Verdict: {verdict}**",'',pd.DataFrame(rows).to_markdown(index=False),'',json.dumps(summary,indent=2)]
    (EXP/'summary/alignn_transfer_report.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['A'],required=True);args=parser.parse_args()
    phase_a()
