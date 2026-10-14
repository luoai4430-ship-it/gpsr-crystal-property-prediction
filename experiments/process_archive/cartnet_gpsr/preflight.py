from __future__ import annotations
import importlib, json, subprocess, sys
from pathlib import Path
import torch, torch_geometric

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from data_cartnet import TASKS, fixed_split, load_records

def main():
    task_status={}
    for task,spec in TASKS.items():
        records=load_records(task);splits=fixed_split(task,records)
        sets=[set(r.cid for r in x) for x in splits]
        disjoint=not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        task_status[task]={'total':len(records),'split_sizes':[len(x) for x in splits],'disjoint':disjoint,
                           'group':spec['group'],'first_ids':[x[0].cid if x else None for x in splits]}
        if not disjoint or sum(map(len,splits))!=len(records):raise RuntimeError(task)
    commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/CartNet'),'rev-parse','HEAD'],text=True).strip()
    result={'official_repo':'https://github.com/imatge-upc/CartNet','official_commit':commit,
            'python':sys.version,'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'official_config':{'cutoff':5.0,'neighbors':'all within radius','hidden_dim':256,'interaction_layers':4,
                               'optimizer':'Adam','lr':1e-3,'scheduler':'OneCycleLR','epochs':500,'batch_size':64,
                               'checkpoint':'minimum validation MAE; test once'},
            'gpsr':{'injection_module':'CartNet.layers[0] output','text_dim':768,'projector':'Linear-GELU-Linear',
                    'node_gate':'Linear-GELU-Linear-Sigmoid','alpha_init':1e-3,'tau':.10},'tasks':task_status}
    (HERE/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
