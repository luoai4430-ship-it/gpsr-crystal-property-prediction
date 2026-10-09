"""Matched ALIGNN Graph/Early/Hard training for one fixed benchmark task."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import dgl
import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GPSR_DATA_ROOT", "/root/autodl-tmp/dataset1/dataset"))
JARVIS_ARCHIVE = Path(os.environ.get("GPSR_JARVIS_ARCHIVE", "/root/autodl-tmp/dataset1/jdft_3d-8-18-2021.json.zip"))
sys.path.insert(0, str(ROOT / "third_party/alignn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alignn.graphs import Graph, compute_bond_cosines
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn_gpsr_model import ALIGNNGPSR, graph_recovery_check


TASKS = {
    "Jarvis-Bandgap_MBJ": {
        "rel": "jarvis/mbj_bandgap",
        "raw": "Jarvis-Bandgap_MBJ-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "jarvis",
        "log10": False,
    },
    "Jarvis-Bandgap_OPT": {
        "rel": "jarvis/optb88vdw_bandgap",
        "raw": "Jarvis-Bandgap_OPT-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "jarvis",
        "log10": False,
    },
    "Jarvis-FormationEnergy": {
        "rel": "jarvis/formation_energy_peratom",
        "raw": "Jarvis-FormationEnergy-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "jarvis",
        "log10": False,
    },
    "Jarvis-TotalEnergy": {
        "rel": "jarvis/optb88vdw_total_energy",
        "raw": "Jarvis-TotalEnergy-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "jarvis",
        "log10": False,
    },
    "Jarvis-BulkModulusKv": {
        "rel": "jarvis/bulk_modulus_kv",
        "raw": "Jarvis-BulkModulusKv-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "jarvis",
        "log10": False,
    },
    "Jarvis-ShearModulusGv": {
        "rel": "jarvis/shear_modulus_gv",
        "raw": "Jarvis-ShearModulusGv-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "jarvis",
        "log10": False,
    },
    "MP-Bandgap": {
        "rel": "mp_2018/MP-bandgap",
        "raw": "MP-Bandgap-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "mp",
        "log10": False,
    },
    "MP-FormationEnergy": {
        "rel": "mp_2018/MP-formation",
        "raw": "MP-FormationEnergy-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "mp",
        "log10": False,
    },
    "MP-BulkModuli": {
        "rel": "mp_2018_small/bulk",
        "raw": "MP-BulkModuli-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "mp",
        "log10": True,
    },
    "MP-ShearModuli": {
        "rel": "mp_2018_small/shear",
        "raw": "MP-ShearModuli-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt",
        "family": "mp",
        "log10": True,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Record:
    cid: str
    y: float
    text: torch.Tensor
    g: dgl.DGLGraph | None = None
    lg: dgl.DGLGraph | None = None
    lattice: torch.Tensor | None = None


class FixedGraphDataset(Dataset):
    def __init__(self, records: List[Record], mean: float, std: float):
        self.records = records
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return r.g, r.lg, r.lattice, (r.y - self.mean) / self.std, r.text, r.cid


def collate(samples):
    graphs, line_graphs, lattices, labels, texts, cids = map(list, zip(*samples))
    return (
        dgl.batch(graphs),
        dgl.batch(line_graphs),
        torch.stack(lattices),
        torch.tensor(labels, dtype=torch.float32),
        torch.stack(texts).float(),
        list(cids),
    )


def load_records(task: str) -> List[Record]:
    spec = TASKS[task]
    raw_path = ROOT / "Dataset/raw" / spec["raw"]
    if raw_path.exists():
        raw = torch.load(raw_path, map_location="cpu")
        records = [
            Record(
                cid=str(item.cid),
                y=float(item.y),
                text=item.text_emb.float().reshape(-1),
            )
            for item in raw
        ]
    else:
        # Phase B aliases share audited text caches but may not have a task-specific raw file.
        task_root = DATA_ROOT / spec["rel"]
        frame = pd.read_csv(task_root / "cif/id_prop.csv", header=None)
        embeddings = torch.load(task_root / "MatBert_embeddings.pt", map_location="cpu")
        records = []
        for cid, y in frame.iloc[:, :2].itertuples(index=False, name=None):
            value = float(y)
            if spec["log10"]:
                value = math.log10(max(value, 1e-30))
            entry = embeddings.get(cid, embeddings.get(str(cid)))
            if isinstance(entry, dict):
                entry = next(entry[k] for k in ("embedding", "vec", "features") if k in entry)
            records.append(Record(str(cid), value, torch.as_tensor(entry).float().reshape(-1)))
    random.Random(123).shuffle(records)
    return records


def split_records(task: str, records: List[Record]):
    spec = TASKS[task]
    n = len(records)
    if spec["family"] == "jarvis" or n < 65000:
        n_train, n_valid = int(0.8 * n), int(0.1 * n)
    else:
        n_train, n_valid = 60000, 5000
    return records[:n_train], records[n_train:n_train+n_valid], records[n_train+n_valid:]


def build_graphs(task: str, records: List[Record], log_path: Path) -> None:
    cif_dir = DATA_ROOT / TASKS[task]["rel"] / "cif"
    atoms_by_id = None
    if TASKS[task]["family"] == "jarvis":
        archive = JARVIS_ARCHIVE
        with zipfile.ZipFile(archive) as handle:
            with handle.open(handle.namelist()[0]) as stream:
                official = json.load(stream)
        def structure_id(cid: str) -> str:
            return cid.split("__record_", 1)[0]

        wanted = {structure_id(record.cid) for record in records}
        official_by_id = {
            str(item["jid"]): item["atoms"]
            for item in official
            if str(item["jid"]) in wanted
        }
        if len(official_by_id) != len(wanted):
            missing = sorted(wanted - official_by_id.keys())[:20]
            raise RuntimeError(f"missing JARVIS structures: {missing}")
        atoms_by_id = {
            record.cid: official_by_id[structure_id(record.cid)]
            for record in records
        }
        del official
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        for idx, record in enumerate(tqdm(records, desc=f"graphs:{task}")):
            if atoms_by_id is not None:
                atoms = Atoms.from_dict(atoms_by_id[record.cid])
            else:
                cif_path = cif_dir / f"{record.cid}.cif"
                atoms = Atoms.from_cif(str(cif_path), use_cif2cell=False)
            g, lg = Graph.atom_dgl_multigraph(
                atoms=atoms,
                neighbor_strategy="k-nearest",
                cutoff=8.0,
                max_neighbors=12,
                atom_features="cgcnn",
                compute_line_graph=True,
                use_canonize=True,
            )
            record.g = g
            record.lg = lg
            record.lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
            if (idx + 1) % 1000 == 0 or idx + 1 == len(records):
                elapsed = time.time() - started
                line = json.dumps({
                    "event": "graph_progress", "task": task, "done": idx + 1,
                    "total": len(records), "elapsed_s": elapsed,
                })
                print(line, flush=True)
                log.write(line + "\n"); log.flush()


def make_loaders(task: str, records: List[Record], batch_size: int, seed: int):
    train, valid, test = split_records(task, records)
    train_values = torch.tensor([r.y for r in train], dtype=torch.float32)
    mean = float(train_values.mean())
    std = float(train_values.std())
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=collate)
    train_loader = DataLoader(FixedGraphDataset(train, mean, std), shuffle=True, drop_last=True, generator=generator, **kwargs)
    valid_loader = DataLoader(FixedGraphDataset(valid, mean, std), shuffle=False, drop_last=True, **kwargs)
    test_loader = DataLoader(FixedGraphDataset(test, mean, std), shuffle=False, drop_last=False, **kwargs)
    return train_loader, valid_loader, test_loader, mean, std, (len(train), len(valid), len(test))


def graph_model() -> ALIGNN:
    return ALIGNN(ALIGNNConfig(
        name="alignn", alignn_layers=4, gcn_layers=4,
        atom_input_features=92, edge_input_features=80,
        triplet_input_features=40, embedding_features=64,
        hidden_features=256, output_features=1,
    ))


def move_batch(batch, device):
    g, lg, lattice, y, text, cids = batch
    return g.to(device), lg.to(device), lattice.to(device), y.to(device), text.to(device), cids


@torch.no_grad()
def evaluate(model, loader, mean, std, device, semantic: bool, diagnostics: bool = False):
    model.eval()
    rows = []
    diag: Dict[str, List[float]] = {}
    total_abs = 0.0
    total_n = 0
    for batch in loader:
        g, lg, lattice, y, text, cids = move_batch(batch, device)
        if semantic:
            pred_norm, info = model((g, lg, lattice), text, collect_diagnostics=diagnostics)
        else:
            pred_norm = model((g, lg, lattice))
            info = {}
        pred_norm = pred_norm.reshape(-1)
        true = y * std + mean
        pred = pred_norm * std + mean
        total_abs += float((pred - true).abs().sum())
        total_n += int(true.numel())
        rows.extend(zip(cids, true.detach().cpu().tolist(), pred.detach().cpu().tolist()))
        if diagnostics:
            for key, value in info.items():
                if key == "delta_final":
                    continue
                diag.setdefault(key, []).extend(value.detach().float().cpu().reshape(-1).tolist())
    return total_abs / total_n, rows, diag


def summarize_diagnostics(diag: Dict[str, List[float]], model: ALIGNNGPSR) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, values in diag.items():
        a = np.asarray(values, dtype=float)
        out[f"mean_{key}"] = float(np.mean(a))
        out[f"median_{key}"] = float(np.median(a))
        out[f"p90_{key}"] = float(np.quantile(a, 0.9))
        out[f"max_{key}"] = float(np.max(a))
    rho = np.asarray(diag["rho_raw"])
    out["clipped_sample_ratio"] = float(np.mean(rho > model.tau))
    out["alpha_mean_abs"] = float(model.alpha.detach().abs().mean().cpu())
    out["alpha_max_abs"] = float(model.alpha.detach().abs().max().cpu())
    return out


def train_one(task: str, mode: str, records: List[Record], args, task_dir: Path):
    out = task_dir / mode
    out.mkdir(parents=True, exist_ok=True)
    complete = out / "metrics.json"
    if complete.exists():
        print(f"SKIP complete {task}/{mode}", flush=True)
        return json.loads(complete.read_text())

    set_seed(args.seed)
    device = torch.device("cuda")
    train_loader, valid_loader, test_loader, mean, std, split_sizes = make_loaders(
        task, records, args.batch_size, args.seed
    )
    semantic = mode != "graph"
    if mode == "graph":
        model = graph_model()
        recovery_error = None
        alpha_init = None
    else:
        model = ALIGNNGPSR("hard" if mode == "hard" else "early", tau=0.10)
        alpha_init = float(model.alpha.detach().mean())
        recovery_error = graph_recovery_check(model, torch.device("cpu"))
        if recovery_error != 0.0:
            raise RuntimeError(f"graph recovery failed: {recovery_error}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3,
    )
    criterion = nn.L1Loss()
    best_valid = float("inf")
    best_epoch = -1
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            g, lg, lattice, y, text, _cids = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if semantic:
                pred, _ = model((g, lg, lattice), text)
            else:
                pred = model((g, lg, lattice))
            loss = criterion(pred.reshape(-1), y.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += float(loss.detach())
            n_batches += 1
        valid_mae, _, _ = evaluate(model, valid_loader, mean, std, device, semantic)
        row = {
            "epoch": epoch + 1,
            "train_loss_normalized": train_loss / n_batches,
            "valid_mae": valid_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        print(json.dumps({"event": "epoch", "task": task, "mode": mode, **row}), flush=True)
        pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
        if valid_mae < best_valid:
            best_valid = valid_mae
            best_epoch = epoch + 1
            torch.save({
                "model": model.state_dict(), "epoch": best_epoch,
                "valid_mae": best_valid, "mean": mean, "std": std,
            }, out / "best_valid.pth")

    checkpoint = torch.load(out / "best_valid.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_mae, predictions, diag = evaluate(
        model, test_loader, mean, std, device, semantic, diagnostics=semantic
    )
    pd.DataFrame(predictions, columns=["cid", "target", "prediction"]).to_csv(
        out / "test_predictions.csv", index=False
    )
    metrics = {
        "task": task, "model": mode, "seed": args.seed,
        "split_seed": 123, "epochs": args.epochs, "batch_size": args.batch_size,
        "best_epoch": best_epoch, "valid_mae": best_valid, "test_mae": test_mae,
        "train_size": split_sizes[0], "valid_size": split_sizes[1], "test_size": split_sizes[2],
        "target_mean": mean, "target_std": std,
        "graph_recovery_max_abs_error": recovery_error,
        "alpha_initial_mean": alpha_init,
    }
    if semantic:
        metrics.update(summarize_diagnostics(diag, model))
        if mode == "hard" and metrics["max_rho_final"] > 0.1001:
            raise RuntimeError(f"hard bound violated: {metrics['max_rho_final']}")
    complete.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", **metrics}), flush=True)
    del model, optimizer, scheduler, train_loader, valid_loader, test_loader
    gc.collect(); torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--models", nargs="+", default=["graph", "early", "hard"], choices=["graph", "early", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    task_dir = out_root / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    run_log = task_dir / "run.log"
    records = load_records(args.task)
    if args.smoke:
        records = records[:640]
        args.epochs = 1
    print(json.dumps({"event": "loaded", "task": args.task, "samples": len(records)}), flush=True)
    build_graphs(args.task, records, run_log)
    for mode in args.models:
        train_one(args.task, mode, records, args, task_dir)


if __name__ == "__main__":
    main()
