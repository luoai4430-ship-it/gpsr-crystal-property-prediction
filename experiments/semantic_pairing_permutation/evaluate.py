"""Post-hoc semantic-pairing permutation control for a frozen GPSR checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run as run_module
from data.dataset import get_dataloader
from model.semantic_residual import as_text_matrix
from train_and_test import build_model


def load_state(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
        return obj
    return getattr(obj, "model_para", obj)


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(n)
    for _ in range(1000):
        perm = rng.permutation(n)
        if np.all(perm != base):
            return perm
    # Deterministic fallback; valid for n > 1.
    return np.roll(base, 1)


def configure(args):
    ns = SimpleNamespace(
        task="train_and_test", target=args.target,
        model="gpsrnodegate-early-bounded", use_cif_csv=True,
        cif_dir=args.cif_dir, text_emb_path=args.text_emb_path,
        num_epoch=60, batch_size=args.batch_size, seed=args.seed,
        lr=1e-3, max_lr=1e-3, weight_decay=1e-4,
        alpha_init=None, alpha_max=.2, lambda_delta=0.,
        max_residual_ratio=.1, lambda_res=0., rho_target=.1,
        sample_gate_bias=3., inject_position="early", inject_after=0,
        text_shuffle_seed=-1, share_text_cache=False,
    )
    run_module.configure(ns)


def collect_text_and_ids(loader):
    texts, ids = [], []
    for batch in loader:
        t = as_text_matrix(batch.text_emb, int(batch.num_graphs)).detach().cpu()
        texts.append(t)
        cid = getattr(batch, "cid", None)
        ids.extend([str(x) for x in cid] if cid is not None else
                   [str(len(ids) + i) for i in range(int(batch.num_graphs))])
    return torch.cat(texts, dim=0), ids


def predict(model, loader, mean, std, replacement=None):
    preds, truth = [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            b = int(batch.num_graphs)
            batch = batch.cuda()
            if replacement is not None:
                batch.text_emb = replacement[offset:offset + b].to(batch.x.device)
            out = model(batch)["property_pred"].reshape(-1)
            preds.extend((out.detach().cpu().numpy() * std + mean).tolist())
            truth.extend((batch.y.detach().cpu().numpy().reshape(-1) * std + mean).tolist())
            offset += b
    if replacement is not None and offset != len(replacement):
        raise RuntimeError(f"replacement rows used={offset}, available={len(replacement)}")
    return np.asarray(preds, float), np.asarray(truth, float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--cif-dir", required=True)
    p.add_argument("--text-emb-path", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--permutations", type=int, default=20)
    p.add_argument("--permutation-seed", type=int, default=20261009)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    configure(args)
    model = build_model().cuda().eval()
    model.load_state_dict(load_state(args.checkpoint))
    _, _, loader, mean, std = get_dataloader()
    all_text, ids = collect_text_and_ids(loader)
    if len(all_text) < 2:
        raise RuntimeError("Permutation control requires at least two test samples")

    correct, truth = predict(model, loader, mean, std)
    formal_files = list(args.run_dir.glob("*_test_pred_true.csv"))
    if len(formal_files) != 1:
        raise RuntimeError(f"expected one formal prediction CSV, got {formal_files}")
    formal = pd.read_csv(formal_files[0])
    formal_pred = formal["y_pred"].to_numpy(float)
    formal_true = formal["y_true"].to_numpy(float)
    if len(formal) != len(correct):
        raise RuntimeError("formal/recomputed row mismatch")
    max_pred_diff = float(np.max(np.abs(correct - formal_pred)))
    # The model predicts standardized targets in float32.  Comparing after
    # multiplying by a large target standard deviation can turn a sub-1e-6
    # native discrepancy from CUDA scatter reductions into >1e-5 physical
    # units.  Audit both scales and enforce the tolerance in model-output
    # (standardized) units.
    max_pred_diff_standardized = float(max_pred_diff / max(abs(float(std)), 1e-12))
    max_true_diff = float(np.max(np.abs(truth - formal_true)))
    if max_pred_diff_standardized > 1e-5 or max_true_diff > 1e-5:
        raise RuntimeError(
            "formal reproduction failed "
            f"pred_raw={max_pred_diff} pred_standardized={max_pred_diff_standardized} "
            f"true={max_true_diff}"
        )

    with torch.no_grad():
        alpha = model.injector.alpha.detach().clone()
        model.injector.alpha.zero_()
    off, truth_off = predict(model, loader, mean, std)
    with torch.no_grad():
        model.injector.alpha.copy_(alpha)
    if not np.allclose(truth, truth_off, rtol=0, atol=1e-7):
        raise RuntimeError("truth changed in semantic-OFF pass")

    rng = np.random.default_rng(args.permutation_seed + args.seed)
    perm_rows = []
    shuffled_preds = []
    for i in range(args.permutations):
        perm = derangement(len(all_text), rng)
        pred, truth_perm = predict(model, loader, mean, std, all_text[perm])
        if not np.allclose(truth, truth_perm, rtol=0, atol=1e-7):
            raise RuntimeError(f"truth changed in permutation {i}")
        shuffled_preds.append(pred)
        mae = float(np.mean(np.abs(pred - truth)))
        perm_rows.append({
            "task": args.target, "seed": args.seed, "permutation_index": i,
            "test_mae": mae, "fixed_point_fraction": float(np.mean(perm == np.arange(len(perm)))),
        })

    shuffled = np.vstack(shuffled_preds)
    correct_err = np.abs(correct - truth)
    off_err = np.abs(off - truth)
    shuffled_err = np.abs(shuffled - truth[None, :])
    correct_mae = float(correct_err.mean())
    off_mae = float(off_err.mean())
    shuffled_mae = shuffled_err.mean(axis=1)
    p_value = float((1 + np.sum(shuffled_mae <= correct_mae)) / (1 + len(shuffled_mae)))

    sample = pd.DataFrame({
        "task": args.target, "seed": args.seed, "sample_id": ids,
        "y_true": truth, "y_correct": correct, "y_off": off,
        "correct_abs_error": correct_err, "off_abs_error": off_err,
        "shuffle_mean_prediction": shuffled.mean(axis=0),
        "shuffle_prediction_sd": shuffled.std(axis=0, ddof=1) if len(shuffled) > 1 else 0.,
        "shuffle_mean_abs_error": shuffled_err.mean(axis=0),
        "pairing_error_gain": shuffled_err.mean(axis=0) - correct_err,
        "correct_better_than_shuffle_fraction": (correct_err[None, :] < shuffled_err).mean(axis=0),
    })
    sample.to_csv(args.out / "sample_metrics.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(args.out / "permutation_metrics.csv", index=False)
    np.savez_compressed(args.out / "predictions.npz", correct=correct, off=off,
                        shuffled=shuffled, truth=truth)

    summary = {
        "task": args.target, "seed": args.seed, "num_samples": int(len(truth)),
        "num_permutations": int(args.permutations), "correct_mae": correct_mae,
        "off_mae": off_mae, "shuffle_mean_mae": float(shuffled_mae.mean()),
        "shuffle_sample_sd_mae": float(shuffled_mae.std(ddof=1)) if len(shuffled_mae) > 1 else 0.,
        "pairing_advantage_pct_vs_shuffle": float(100 * (1 - correct_mae / shuffled_mae.mean())),
        "correct_gain_pct_vs_off": float(100 * (1 - correct_mae / off_mae)),
        "permutation_p_one_sided": p_value,
        "fraction_samples_pairing_helped": float((sample.pairing_error_gain > 0).mean()),
        "formal_reproduction_max_abs_pred_diff": max_pred_diff,
        "formal_reproduction_max_abs_pred_diff_standardized": max_pred_diff_standardized,
        "formal_reproduction_max_abs_true_diff": max_true_diff,
        "control_note": "Frozen validation-selected checkpoint; test embeddings permuted without replacement; no training or model selection.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
