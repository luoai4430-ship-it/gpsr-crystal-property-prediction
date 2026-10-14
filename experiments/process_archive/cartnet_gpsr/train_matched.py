"""Train one matched CartNet or CartNet+GPSR run on the fixed benchmark."""
from __future__ import annotations

import argparse
import copy
import csv
import gc
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
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CARTNET = ROOT / "third_party/CartNet"
sys.path.insert(0, str(CARTNET))
sys.path.insert(0, str(HERE))

from models.cartnet import CartNet
from cartnet_gpsr_model import CartNetGPSR
from data_cartnet import GeometryStore, TASKS, TaskDataset, fixed_split, load_records, write_and_check_manifest


def configure():
    set_cfg(cfg)
    cfg.radius = 5.0
    cfg.invariant = False
    cfg.use_temp = False
    cfg.use_atom_types = True
    cfg.envelope = True
    cfg.dim_in = 256
    cfg.dim_rbf = 64
    cfg.num_layers = 4


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); seed_everything(seed)


def make_model(mode: str):
    kwargs = dict(dim_in=cfg.dim_in, dim_rbf=cfg.dim_rbf, num_layers=cfg.num_layers,
                  radius=cfg.radius, invariant=cfg.invariant, temperature=False,
                  use_envelope=True, atom_types=True, cholesky=False)
    if mode == "graph":
        return CartNet(**kwargs)
    if mode == "gpsr":
        return CartNetGPSR(tau=0.10, **kwargs)
    raise ValueError(mode)


def unpack_output(model, batch, diagnostics=False):
    if isinstance(model, CartNetGPSR):
        pred, true, info = model(batch, collect_diagnostics=diagnostics)
    else:
        pred, true = model(batch); info = {}
    return pred.reshape(-1), true.reshape(-1), info


@torch.no_grad()
def evaluate(model, loader, device, diagnostics=False):
    model.eval()
    absolute = 0.0; count = 0; rows = []; diag = {}
    for batch in loader:
        batch = batch.to(device)
        pred, true, info = unpack_output(model, batch, diagnostics)
        absolute += float((pred - true).abs().sum())
        count += int(true.numel())
        cids = list(batch.cid)
        rows.extend(zip(cids, true.detach().cpu().tolist(), pred.detach().cpu().tolist()))
        if diagnostics:
            for key, value in info.items():
                if key == "delta_final":
                    continue
                diag.setdefault(key, []).extend(value.detach().float().cpu().reshape(-1).tolist())
    return absolute / max(count, 1), rows, diag


def summarize_diag(diag, model):
    out = {}
    for key, values in diag.items():
        a = np.asarray(values, dtype=float)
        if a.size:
            out[f"mean_{key}"] = float(a.mean())
            out[f"median_{key}"] = float(np.median(a))
            out[f"p90_{key}"] = float(np.quantile(a, .9))
            out[f"max_{key}"] = float(a.max())
    out["alpha_mean_abs"] = float(model.alpha.detach().abs().mean())
    out["text_projector_parameter_norm"] = float(torch.linalg.vector_norm(
        torch.cat([p.detach().reshape(-1) for p in model.text_projector.parameters()])
    ))
    return out


@torch.no_grad()
def equivalence_check(loader, device):
    set_seed(271828)
    graph = make_model("graph").to(device).eval()
    set_seed(271828)
    gpsr = make_model("gpsr").to(device).eval()
    missing, unexpected = gpsr.load_state_dict(graph.state_dict(), strict=False)
    assert all(k.startswith(("text_projector", "node_gate", "alpha")) for k in missing)
    assert not unexpected
    batch = next(iter(loader))
    b1 = copy.deepcopy(batch).to(device); b2 = copy.deepcopy(batch).to(device)
    graph_pred, _ = graph(b1)
    gpsr.semantic_enabled = False
    gpsr_pred, _, _ = gpsr(b2)
    b3 = copy.deepcopy(batch).to(device)
    before, after = gpsr.first_block_state(b3, semantic=False)
    output_error = float((graph_pred - gpsr_pred).abs().max())
    state_error = float((before - after).abs().max())
    if output_error >= 1e-7 or state_error >= 1e-7:
        raise RuntimeError(f"graph recovery failed: output={output_error}, state={state_error}")
    return {"max_output_abs_error": output_error, "max_atom_state_abs_error": state_error,
            "tolerance": 1e-7, "pass": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--mode", choices=("graph", "gpsr"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure(); set_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.task)
    splits = fixed_split(args.task, records)
    manifest_status = write_and_check_manifest(args.task, splits, out / "split_manifest.csv")
    geometry = GeometryStore(TASKS[args.task]["group"])
    datasets = [TaskDataset(s, geometry) for s in splits]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True,
                              drop_last=False, num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0, generator=generator)
    valid_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=args.workers, pin_memory=True,
                             persistent_workers=args.workers > 0)
    device = torch.device("cuda:0")
    equivalence = equivalence_check(valid_loader, device)
    (out / "graph_path_equivalence.json").write_text(json.dumps(equivalence, indent=2))
    set_seed(args.seed)
    model = make_model(args.mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = args.epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=steps, pct_start=0.01)
    best_val = math.inf; best_epoch = -1; history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train(); train_abs = 0.0; train_n = 0
        for batch in train_loader:
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            pred, true, _ = unpack_output(model, batch)
            loss = F.l1_loss(pred, true)
            loss.backward(); optimizer.step(); scheduler.step()
            train_abs += float((pred.detach() - true).abs().sum()); train_n += int(true.numel())
        val_mae, _, _ = evaluate(model, valid_loader, device)
        row = {"epoch": epoch, "train_mae": train_abs/train_n, "valid_mae": val_mae,
               "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time()-started}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_mae < best_val:
            best_val = val_mae; best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "valid_mae": val_mae, "args": vars(args)}, out / "best_validation.pt")
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    checkpoint = torch.load(out / "best_validation.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_mae, rows, diag = evaluate(model, test_loader, device, diagnostics=args.mode == "gpsr")
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("material_id", "target", "prediction")); writer.writerows(rows)
    metrics = {"task": args.task, "category": TASKS[args.task]["category"], "model": args.mode,
               "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
               "optimizer": "Adam", "scheduler": "OneCycleLR", "learning_rate": 1e-3,
               "checkpoint_rule": "minimum validation MAE; test evaluated once",
               "best_epoch": best_epoch, "best_valid_mae": best_val, "test_mae": test_mae,
               "split_seed": 123, "split_status": manifest_status,
               "graph": {"cutoff": 5.0, "neighbors": "all within radius", "directional": True},
               "elapsed_s": time.time()-started, "graph_path_equivalence": equivalence}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if args.mode == "gpsr":
        diag_summary = summarize_diag(diag, model)
        (out / "diagnostics.json").write_text(json.dumps(diag_summary, indent=2))
        if diag_summary.get("max_rho_final", math.inf) > 0.1001:
            raise RuntimeError(f"hard bound violation: {diag_summary['max_rho_final']}")
    print(json.dumps({"event": "complete", **metrics}), flush=True)
    del model, geometry, train_loader, valid_loader, test_loader
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
