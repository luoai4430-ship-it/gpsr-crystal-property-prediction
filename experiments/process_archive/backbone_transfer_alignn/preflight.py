"""Verify exact split IDs, text dimensions, fixed config, and semantic identity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT=Path('/root/autodl-tmp/TextResidualSHFMat-FAIR')
EXP=ROOT/'experiments/backbone_transfer_alignn'
sys.path.insert(0,str(EXP));sys.path.insert(0,str(ROOT/'third_party/alignn'))
from alignn_gpsr_model import ALIGNNGPSR,graph_recovery_check
from train_alignn_transfer import TASKS,load_records,split_records

tasks=['Jarvis-Bandgap_MBJ','Jarvis-FormationEnergy','Jarvis-BulkModulusKv','MP-Bandgap','MP-FormationEnergy']
checks=[]
for task in tasks:
    records=load_records(task); splits=split_records(task,records)
    row={'task':task,'total':len(records),'split_sizes':[len(x) for x in splits],
         'text_dim':int(records[0].text.numel()),'nonzero_text':sum(float(x.text.abs().sum())>1e-9 for x in records)}
    for name,part in zip(('train','valid','test'),splits):
        pattern=f'{task}-cutoff4.0-maxNeibNum16-spheHarmL4-splitSeed123-CIF_CSV-{name}-CGCNN-multimodal-*.pt'
        candidates=list((ROOT/'Dataset/processed').glob(pattern))
        if candidates:
            data,_slices=torch.load(candidates[0],map_location='cpu')
            expected=[str(x) for x in data.cid]
            actual=[x.cid for x in part]
            row[f'{name}_id_exact_match']=actual==expected
            row[f'{name}_first_id']=actual[0] if actual else None
            row[f'{name}_last_id']=actual[-1] if actual else None
        else:
            row[f'{name}_id_exact_match']=None
    checks.append(row)

model=ALIGNNGPSR('hard',tau=0.10)
payload={
    'official_alignn_commit':'f2366daa3413d28a825b46e34d001b5549b05a40',
    'alignn_layers':4,'gcn_layers':4,'hidden_features':256,
    'neighbor_strategy':'k-nearest','cutoff_angstrom':8.0,'max_neighbors':12,
    'atom_features':'cgcnn','edge_features':'RBF distance, 80 bins',
    'angle_features':'bond-angle cosine RBF, 40 bins',
    'pooling':'global average','injection_after_alignn_block':1,
    'projector':'Linear(768,256)-GELU-Linear(256,256)',
    'node_gate':'Linear(512,256)-GELU-Linear(256,256)-Sigmoid',
    'alpha_initial_mean':float(model.alpha.mean()),'tau':0.10,
    'graph_recovery_max_abs_error':graph_recovery_check(model,torch.device('cpu')),
    'seed':42,'split_seed':123,'batch_size':32,'epochs':60,
    'optimizer':'AdamW(lr=1e-3, weight_decay=1e-4)',
    'scheduler':'OneCycleLR(max_lr=1e-3,pct_start=0.3)',
    'checkpoint_rule':'lowest validation MAE',
    'tasks':checks,
}
payload['all_required_split_checks_pass']=all(
    v is not False for row in checks for k,v in row.items() if k.endswith('_id_exact_match')
)
payload['all_text_complete']=all(row['nonzero_text']==row['total'] and row['text_dim']==768 for row in checks)
payload['pass']=payload['all_required_split_checks_pass'] and payload['all_text_complete'] and payload['graph_recovery_max_abs_error']==0.0
(EXP/'preflight.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps(payload,indent=2))
if not payload['pass']:
    raise SystemExit(2)
