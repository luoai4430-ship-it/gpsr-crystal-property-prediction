"""One counterfactual semantic-OFF pass for a validation-selected checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run as run_module
from data.dataset import get_dataloader
from train_and_test import build_model


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
        return obj
    return getattr(obj, "model_para", obj)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--method", choices=("smm", "unbounded", "gpsr"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cif-dir", required=True)
    p.add_argument("--text-emb-path", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    cli_model = {"smm": "smm", "unbounded": "gpsrnodegate-early",
                 "gpsr": "gpsrnodegate-early-bounded"}[a.method]
    ns = SimpleNamespace(task="train_and_test", target=a.target, model=cli_model,
        use_cif_csv=True, cif_dir=a.cif_dir, text_emb_path=a.text_emb_path,
        num_epoch=60, batch_size=32, seed=a.seed, lr=1e-3, max_lr=1e-3,
        weight_decay=1e-4, alpha_init=None, alpha_max=.2, lambda_delta=0.,
        max_residual_ratio=.1 if a.method == "gpsr" else float("inf"),
        lambda_res=0., rho_target=.1, sample_gate_bias=3., inject_position="early",
        inject_after=0, text_shuffle_seed=-1, share_text_cache=False)
    run_module.configure(ns)
    model = build_model().cuda().eval()
    model.load_state_dict(load_state(a.checkpoint))
    _, _, loader, mean, std = get_dataloader()

    injector = model.injector
    if a.method == "smm":
        injector.semantic_enabled = False
    else:
        with torch.no_grad():
            injector.alpha.zero_()
    off, truth, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.cuda()
            out = model(batch)
            pred = out["property_pred"].reshape(-1)
            off.extend((pred.cpu().numpy() * std + mean).tolist())
            truth.extend((batch.y.cpu().numpy().reshape(-1) * std + mean).tolist())
            cid = getattr(batch, "cid", None)
            ids.extend([str(x) for x in cid] if cid is not None else
                       [str(len(ids) + i) for i in range(int(batch.num_graphs))])

    rd = Path(a.run_dir)
    pred_path = next(rd.glob("*_test_pred_true.csv"))
    on = pd.read_csv(pred_path)
    samples = pd.read_csv(rd / "gpsr_test_sample_stats.csv")
    if not (len(on) == len(samples) == len(off)):
        raise RuntimeError(f"row mismatch on={len(on)} stats={len(samples)} off={len(off)}")
    on_pred = on["y_pred"].to_numpy(float)
    true = on["y_true"].to_numpy(float)
    off = np.asarray(off, float)
    mae_off = float(np.mean(np.abs(off - true)))
    diag = samples.copy()
    diag.insert(0, "task", a.target)
    diag.insert(1, "method", a.method)
    diag.insert(2, "seed", a.seed)
    diag["sample_id"] = ids
    diag["y_true"] = true
    diag["y_on"] = on_pred
    diag["y_off"] = off
    diag["delta_y"] = on_pred - off
    diag["normalized_output_shift"] = np.abs(diag["delta_y"]) / max(mae_off, 1e-12)
    diag.to_csv(rd / "semantic_on_off_diagnostics.csv", index=False)

    rho = diag["rho_final"].to_numpy(float)
    summary = {
        "task": a.target, "method": a.method, "seed": a.seed,
        "mae_on": float(np.mean(np.abs(on_pred - true))), "mae_off": mae_off,
        "mean_rho": float(rho.mean()), "median_rho": float(np.median(rho)),
        "p90_rho": float(np.quantile(rho, .90)), "p95_rho": float(np.quantile(rho, .95)),
        "max_rho": float(rho.max()),
        "mean_cos_h_off_on": float(diag["cos_h_off_on"].mean()),
        "mean_cos_delta_h": float(diag["cos_delta_h"].mean()),
        "mean_abs_delta_y": float(np.mean(np.abs(diag["delta_y"]))),
        "mean_normalized_output_shift": float(diag["normalized_output_shift"].mean()),
        "counterfactual_note": "semantic-OFF diagnostic pass; formal ON test was evaluated once during training",
    }
    (rd / "on_off_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
