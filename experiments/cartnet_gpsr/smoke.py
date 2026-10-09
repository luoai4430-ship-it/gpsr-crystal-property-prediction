import copy,json,os,sys,zipfile
from pathlib import Path
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.specie import get_node_attributes
from torch_geometric.data import Data,Batch
from torch_geometric.graphgym.config import cfg,set_cfg
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'third_party/CartNet'));sys.path.insert(0,str(Path(__file__).parent))
from dataset.utils import radius_graph_pbc
from models.cartnet import CartNet
from cartnet_gpsr_model import CartNetGPSR
set_cfg(cfg);cfg.radius=5.;cfg.invariant=False
archive=os.environ.get('GPSR_JARVIS_ARCHIVE','/root/autodl-tmp/dataset1/jdft_3d-8-18-2021.json.zip')
with zipfile.ZipFile(archive) as z:
 import json as j; row=j.loads(z.read(z.namelist()[0]))[0]
a=Atoms.from_dict(row['atoms']);z=torch.tensor([get_node_attributes(s,atom_features='atomic_number') for s in a.elements]).long().reshape(-1)
d=Data(x=z,y=torch.tensor([0.]));d.pos=torch.tensor(a.cart_coords).float();d.cell=torch.tensor(a.lattice.matrix).float().unsqueeze(0);d.pbc=torch.tensor([[True,True,True]]);d.natoms=torch.tensor([z.numel()])
b=Batch.from_data_list([d]);ei,_,_,v=radius_graph_pbc(b,5.,None);d.edge_index=ei;d.cart_dist=torch.linalg.vector_norm(v,dim=-1);d.cart_dir=torch.nn.functional.normalize(v,p=2,dim=-1);del d.pbc;d.text_emb=torch.randn(1,768)
b=Batch.from_data_list([d]).cuda();kw=dict(dim_in=256,dim_rbf=64,num_layers=4,radius=5.,invariant=False,temperature=False,use_envelope=True,atom_types=True,cholesky=False)
torch.manual_seed(1);g=CartNet(**kw).cuda().eval();torch.manual_seed(1);s=CartNetGPSR(tau=.1,**kw).cuda().eval();miss,unexp=s.load_state_dict(g.state_dict(),strict=False);s.semantic_enabled=False
with torch.no_grad():pg,_=g(copy.deepcopy(b));ps,_,_=s(copy.deepcopy(b));before,after=s.first_block_state(copy.deepcopy(b),False)
assert (pg-ps).abs().max()<1e-7 and (before-after).abs().max()<1e-7
s.semantic_enabled=True;pred,true,info=s(copy.deepcopy(b),collect_diagnostics=True);loss=(pred.reshape(-1)-true.reshape(-1)).abs().mean();loss.backward()
assert float(info['rho_final'].max())<=.1001
print(json.dumps({'pass':True,'output_error':float((pg-ps).abs().max()),'state_error':float((before-after).abs().max()),'rho_final':float(info['rho_final'].max()),'missing':miss,'unexpected':unexp}))
