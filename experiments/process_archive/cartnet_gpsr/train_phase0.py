"""Official CartNet graph-only reproduction anchors (one prescribed seed each)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, set_cfg

ROOT = Path(__file__).resolve().parents[2]
CARTNET = ROOT / "third_party/CartNet"
sys.path.insert(0, str(CARTNET))
from loader.loader import create_loader
from models.master import create_model

TASKS = {
    "JARVIS-MBJ": {"dataset": "jarvis", "target": "mbj_bandgap", "seed": 1, "reference": 0.25303},
    "JARVIS-OPT": {"dataset": "jarvis", "target": "optb88vdw_bandgap", "seed": 1, "reference": 0.11531},
    "MP-Bandgap": {"dataset": "megnet", "target": "gap pbe", "seed": 0, "reference": 0.1908},
    "MP-Bulk": {"dataset": "megnet", "target": "bulk modulus", "seed": 0, "reference": 0.0330},
}


def configure(spec, out):
    set_cfg(cfg)
    cfg.seed = spec["seed"]
    cfg.dataset.task_type = "regression"
    cfg.batch = 64; cfg.batch_accumulation = 1
    cfg.dataset.name = spec["dataset"]
    cfg.dataset_path = str(ROOT / "experiments/cartnet_gpsr/phase0_official_reproduction/official_processed")
    cfg.figshare_target = spec["target"]
    cfg.loss = "MAE"; cfg.optim.max_epoch = 500
    cfg.lr = 1e-3; cfg.warmup = 0.01
    cfg.model = "CartNet"; cfg.max_neighbours = -1
    cfg.radius = 5.0; cfg.num_layers = 4; cfg.dim_in = 256; cfg.dim_rbf = 64
    cfg.augment = False; cfg.invariant = False; cfg.use_temp = False
    cfg.standarize_temp = True; cfg.envelope = True; cfg.use_H = True
    cfg.use_atom_types = True; cfg.workers = 2


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); total = 0.0; count = 0; rows = []
    for batch in loader:
        batch = batch.to("cuda:0")
        pred, true = model(batch); pred = pred.reshape(-1); true = true.reshape(-1)
        total += float((pred - true).abs().sum()); count += int(true.numel())
        rows.extend(zip(true.cpu().tolist(), pred.cpu().tolist()))
    return total / count, rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=500)
    args = p.parse_args(); spec = TASKS[args.task]; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(spec["seed"]); np.random.seed(spec["seed"]); torch.manual_seed(spec["seed"])
    torch.cuda.manual_seed_all(spec["seed"]); seed_everything(spec["seed"])
    configure(spec, out); cfg.optim.max_epoch = args.epochs
    # The official MEGNet mechanical-property loader resolves its published
    # pickle files relative to the CartNet repository root.
    os.chdir(CARTNET)
    loaders = create_loader(); model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3,
                           total_steps=args.epochs * len(loaders[0]), pct_start=0.01)
    best = math.inf; best_epoch = -1; history = []; started = time.time()
    for epoch in range(args.epochs):
        model.train(); abs_sum = 0.0; n = 0
        for batch in loaders[0]:
            batch = batch.to("cuda:0"); optimizer.zero_grad(set_to_none=True)
            pred, true = model(batch); pred = pred.reshape(-1); true = true.reshape(-1)
            loss = F.l1_loss(pred, true); loss.backward(); optimizer.step(); scheduler.step()
            abs_sum += float((pred.detach()-true).abs().sum()); n += int(true.numel())
        val, _ = evaluate(model, loaders[1])
        row = {"epoch": epoch, "train_mae": abs_sum/n, "valid_mae": val,
               "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time()-started}
        history.append(row); print(json.dumps(row), flush=True)
        if val < best:
            best = val; best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "valid_mae": val}, out/"best_validation.pt")
        with (out/"history.csv").open("w", newline="") as h:
            w=csv.DictWriter(h,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
    ckpt=torch.load(out/"best_validation.pt",map_location="cuda:0");model.load_state_dict(ckpt["model_state"])
    test, rows = evaluate(model, loaders[2])
    with (out/"test_predictions.csv").open("w",newline="") as h:
        w=csv.writer(h);w.writerow(("target","prediction"));w.writerows(rows)
    relative_error=abs(test-spec["reference"])/spec["reference"]
    result={"task":args.task,"dataset":spec["dataset"],"target":spec["target"],"seed":spec["seed"],
            "epochs":args.epochs,"batch_size":64,"optimizer":"Adam","scheduler":"OneCycleLR",
            "cutoff":5.0,"neighbors":"all within radius","hidden_dim":256,"interaction_layers":4,
            "best_epoch":best_epoch,"best_valid_mae":best,"test_mae":test,
            "published_reference":spec["reference"],"relative_error":relative_error,
            "sanity_pass":relative_error<0.10,"checkpoint_rule":"minimum validation MAE; test once",
            "elapsed_s":time.time()-started}
    (out/"metrics.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"event":"complete",**result}),flush=True)


if __name__ == "__main__": main()
