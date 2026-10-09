"""Post-hoc residual-direction analysis for frozen CartNet-GPSR checkpoints.

This script never trains or updates a parameter. It evaluates the fixed seed-42
minimum-validation-MAE checkpoints with the semantic residual disabled/enabled
inside the same trained GPSR model and differentiates only the residual-OFF
prediction with respect to the frozen injection state.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cartnet_gpsr"
HERE = EXP / "posthoc_direction"
CARTNET = ROOT / "third_party/CartNet"
sys.path.insert(0, str(CARTNET))
sys.path.insert(0, str(EXP))

from cartnet_gpsr_model import CartNetGPSR
from data_cartnet import GeometryStore, TASKS, TaskDataset, fixed_split, load_records
from train_matched import configure, make_model, set_seed


ANALYSES = {
    "J_MBJ": {
        "task": "Jarvis-Bandgap_MBJ",
        "label": "J-MBJ",
        "checkpoint": EXP / "phase1_matched_10task/Jarvis-Bandgap_MBJ/gpsr/best_validation.pt",
        "formal_graph_mae": 0.2477434583754167,
        "formal_gpsr_mae": 0.23250933632467707,
    },
    "J_Bulk": {
        "task": "Jarvis-BulkModulusKv",
        "label": "J-Bulk",
        "checkpoint": EXP / "phase1_matched_10task/Jarvis-BulkModulusKv/gpsr/best_validation.pt",
        "formal_graph_mae": 8.788754626018244,
        "formal_gpsr_mae": 8.603873671554938,
    },
    "MP_Bulk": {
        "task": "MP-BulkModuli",
        "label": "MP-Bulk",
        "checkpoint": EXP / "phase1_matched_10task/MP-BulkModuli/gpsr/best_validation.pt",
        "formal_graph_mae": 0.030243743754169338,
        "formal_gpsr_mae": 0.03399093400552792,
    },
}

SAMPLE_METRICS = [
    "rho_raw",
    "rho_final",
    "projection_scale",
    "cos_residual_state",
    "cos_state_update",
    "error_gain",
    "normalized_error_gain",
    "delta_y",
    "target_directed_shift",
    "gradient_alignment",
    "helpful_alignment",
    "downstream_sensitivity",
    "gate_mean",
    "gate_std",
]


def sample_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def downstream(model: CartNetGPSR, base, x: torch.Tensor):
    work = base.clone()
    work.x = x
    work.edge_attr = base.edge_attr.clone()
    for layer in list(model.layers)[1:]:
        work = layer(work)
    pred, true = model.head(work)
    return pred.reshape(-1), true.reshape(-1)


def semantic_tensors(model: CartNetGPSR, h, text, batch_index):
    n = int(text.shape[0])
    z = model.text_projector(text.float())
    z_node = z[batch_index]
    gate = model.node_gate(torch.cat((h, z_node), dim=-1))
    delta_raw = model.alpha * gate * z_node
    h_norm = torch.sqrt(sample_sum(h.square().sum(-1), batch_index, n) + 1e-24)
    raw_norm = torch.sqrt(sample_sum(delta_raw.square().sum(-1), batch_index, n) + 1e-24)
    rho_raw = raw_norm / (h_norm + 1e-12)
    scale = torch.minimum(torch.ones_like(rho_raw), torch.full_like(rho_raw, model.tau) / (rho_raw + 1e-12))
    delta = delta_raw * scale[batch_index, None]
    h_prime = h + delta
    final_norm = torch.sqrt(sample_sum(delta.square().sum(-1), batch_index, n) + 1e-24)
    rho_final = final_norm / (h_norm + 1e-12)
    hp_norm = torch.sqrt(sample_sum(h_prime.square().sum(-1), batch_index, n) + 1e-24)
    dot_dh = sample_sum((delta * h).sum(-1), batch_index, n)
    dot_hhp = sample_sum((h * h_prime).sum(-1), batch_index, n)
    return {
        "z": z,
        "gate": gate,
        "delta_raw": delta_raw,
        "delta": delta,
        "h_prime": h_prime,
        "h_norm": h_norm,
        "raw_norm": raw_norm,
        "final_norm": final_norm,
        "rho_raw": rho_raw,
        "rho_final": rho_final,
        "scale": scale,
        "cos_residual_state": dot_dh / (final_norm * h_norm + 1e-12),
        "cos_state_update": dot_hhp / (h_norm * hp_norm + 1e-12),
    }


def per_sample_gate(gate, batch_index, n):
    rows = []
    for i in range(n):
        values = gate[batch_index == i].reshape(-1)
        p = values.clamp(1e-7, 1 - 1e-7)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        rows.append(
            {
                "gate_mean": values.mean(),
                "gate_std": values.std(unbiased=False),
                "gate_min": values.min(),
                "gate_max": values.max(),
                "gate_entropy": entropy,
                "gate_frac_lt_0_1": (values < 0.1).float().mean(),
                "gate_frac_gt_0_9": (values > 0.9).float().mean(),
            }
        )
    return rows


def load_test_dataset(task: str):
    records = load_records(task)
    _, _, test_records = fixed_split(task, records)
    manifest_path = EXP / "phase1_matched_10task" / task / "gpsr" / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    expected = manifest.loc[manifest["split"] == "test", "material_id"].astype(str).tolist()
    actual = [r.cid for r in test_records]
    if expected != actual:
        raise RuntimeError(f"test manifest mismatch for {task}: expected {len(expected)}, got {len(actual)}")
    geometry = GeometryStore(TASKS[task]["group"])
    return TaskDataset(test_records, geometry), geometry


def run_task(key: str, batch_size: int):
    spec = ANALYSES[key]
    task = spec["task"]
    out_csv = HERE / f"sample_metrics_{key}.csv"
    out_state = HERE / f"injection_states_{key}.pt"
    out_scalar = HERE / f"task_scalars_{key}.json"
    HERE.mkdir(parents=True, exist_ok=True)

    configure()
    set_seed(42)
    dataset, geometry = load_test_dataset(task)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda:0")
    model = make_model("gpsr").to(device)
    if not isinstance(model, CartNetGPSR):
        raise TypeError(type(model))
    checkpoint = torch.load(spec["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    state_ids = []
    state_ptr = [0]
    state_h, state_raw, state_delta, state_hp = [], [], [], []
    z_norm_all = []

    for batch_number, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            base = model.encoder(batch)
            base = model.layers[0](base)
            edge = base.edge_attr.detach()
        h = base.x.detach().requires_grad_(True)
        base.x = h
        base.edge_attr = edge
        sem = semantic_tensors(model, h, batch.text_emb.float(), batch.batch)
        pred_off, true = downstream(model, base, h)
        pred_on, _ = downstream(model, base, sem["h_prime"])
        grad_h = torch.autograd.grad(pred_off.sum(), h, retain_graph=False, create_graph=False)[0]

        n = int(pred_off.numel())
        first_order = sample_sum((grad_h * sem["delta"]).sum(-1), batch.batch, n)
        grad_norm = torch.sqrt(sample_sum(grad_h.square().sum(-1), batch.batch, n) + 1e-24)
        grad_dot = first_order
        alignment = grad_dot / (grad_norm * sem["final_norm"] + 1e-12)
        delta_y = pred_on - pred_off
        target_direction = torch.sign(true - pred_off)
        e_off = (true - pred_off).abs()
        e_on = (true - pred_on).abs()
        error_gain = e_off - e_on
        gate_rows = per_sample_gate(sem["gate"], batch.batch, n)

        for i, cid in enumerate(list(batch.cid)):
            mask = batch.batch == i
            state_ids.append(str(cid))
            node_count = int(mask.sum())
            state_ptr.append(state_ptr[-1] + node_count)
            state_h.append(h[mask].detach().cpu())
            state_raw.append(sem["delta_raw"][mask].detach().cpu())
            state_delta.append(sem["delta"][mask].detach().cpu())
            state_hp.append(sem["h_prime"][mask].detach().cpu())
            z_norm_all.append(float(torch.linalg.vector_norm(sem["z"][i]).detach().cpu()))
            row = {
                "material_id": str(cid),
                "target": float(true[i].detach().cpu()),
                "y_hat_off": float(pred_off[i].detach().cpu()),
                "y_hat_on": float(pred_on[i].detach().cpu()),
                "e_off": float(e_off[i].detach().cpu()),
                "e_on": float(e_on[i].detach().cpu()),
                "error_gain": float(error_gain[i].detach().cpu()),
                "delta_y": float(delta_y[i].detach().cpu()),
                "target_direction": float(target_direction[i].detach().cpu()),
                "target_directed_shift": float((target_direction[i] * delta_y[i]).detach().cpu()),
                "first_order_shift": float(first_order[i].detach().cpu()),
                "gradient_alignment": float(alignment[i].detach().cpu()),
                "helpful_alignment": float((target_direction[i] * alignment[i]).detach().cpu()),
                "rho_raw": float(sem["rho_raw"][i].detach().cpu()),
                "rho_final": float(sem["rho_final"][i].detach().cpu()),
                "projection_scale": float(sem["scale"][i].detach().cpu()),
                "clipped": int(sem["scale"][i].detach().cpu() < 1.0),
                "cos_residual_state": float(sem["cos_residual_state"][i].detach().cpu()),
                "cos_state_update": float(sem["cos_state_update"][i].detach().cpu()),
                "h_norm": float(sem["h_norm"][i].detach().cpu()),
                "raw_residual_norm": float(sem["raw_norm"][i].detach().cpu()),
                "final_residual_norm": float(sem["final_norm"][i].detach().cpu()),
                "grad_h_norm": float(grad_norm[i].detach().cpu()),
                "projected_text_norm": z_norm_all[-1],
            }
            row.update({k: float(v.detach().cpu()) for k, v in gate_rows[i].items()})
            rows.append(row)
        if (batch_number + 1) % 10 == 0 or len(rows) == len(dataset):
            print(json.dumps({"task": task, "samples": len(rows), "total": len(dataset)}), flush=True)

    frame = pd.DataFrame(rows)
    mae_off = float(frame["e_off"].mean())
    mae_on = float(frame["e_on"].mean())
    frame["normalized_error_gain"] = frame["error_gain"] / mae_off
    frame["normalized_output_shift"] = frame["delta_y"].abs() / mae_off
    frame["downstream_sensitivity"] = frame["normalized_output_shift"] / (frame["rho_final"] + 1e-12)
    frame.to_csv(out_csv, index=False)

    states = {
        "task": task,
        "checkpoint": str(spec["checkpoint"]),
        "material_id": state_ids,
        "ptr": torch.tensor(state_ptr, dtype=torch.long),
        "H": torch.cat(state_h),
        "DeltaH_raw": torch.cat(state_raw),
        "DeltaH": torch.cat(state_delta),
        "Hprime": torch.cat(state_hp),
    }
    torch.save(states, out_state)

    prior_on = pd.read_csv(spec["checkpoint"].parent / "test_predictions.csv")
    merged = frame[["material_id", "y_hat_on"]].merge(prior_on, on="material_id", how="inner")
    if len(merged) != len(frame):
        raise RuntimeError(f"stored prediction coverage mismatch for {task}")
    max_on_diff = float(np.max(np.abs(merged["y_hat_on"] - merged["prediction"])))
    stored_mae = json.loads((spec["checkpoint"].parent / "metrics.json").read_text())["test_mae"]
    # GPU scatter reductions are not bitwise deterministic across replay batch
    # layouts. Require the aggregate MAE to reproduce tightly and retain a
    # conservative per-sample tolerance that is negligible on the target scale.
    if abs(mae_on - stored_mae) > 2e-6 or max_on_diff > 1e-3:
        raise RuntimeError(
            f"checkpoint replay mismatch for {task}: mae={mae_on} stored={stored_mae} max_pred_diff={max_on_diff}"
        )

    scalar = {
        "task": task,
        "label": spec["label"],
        "N": len(frame),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_mae": float(checkpoint["valid_mae"]),
        "formal_independent_cartnet_mae": spec["formal_graph_mae"],
        "formal_independent_gpsr_mae": spec["formal_gpsr_mae"],
        "within_gpsr_checkpoint_mae_off": mae_off,
        "within_gpsr_checkpoint_mae_on": mae_on,
        "replay_max_abs_prediction_difference": max_on_diff,
        "alpha_mean_abs": float(model.alpha.detach().abs().mean().cpu()),
        "alpha_l2_norm": float(torch.linalg.vector_norm(model.alpha.detach()).cpu()),
        "mean_projected_text_norm": float(np.mean(z_norm_all)),
        "mean_raw_residual_norm": float(frame["raw_residual_norm"].mean()),
        "mean_final_residual_norm": float(frame["final_residual_norm"].mean()),
        "fraction_improved": float((frame["error_gain"] > 1e-10).mean()),
        "fraction_hurt": float((frame["error_gain"] < -1e-10).mean()),
        "fraction_tied": float((frame["error_gain"].abs() <= 1e-10).mean()),
        "max_rho_final": float(frame["rho_final"].max()),
    }
    out_scalar.write_text(json.dumps(scalar, indent=2), encoding="utf-8")
    print(json.dumps({"event": "posthoc_task_complete", **scalar}), flush=True)


def safe_corr(x, y, kind):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(pearsonr(x, y).statistic if kind == "pearson" else spearmanr(x, y).statistic)


def describe(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {
        "N": len(a),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "Q1": float(np.quantile(a, 0.25)),
        "Q3": float(np.quantile(a, 0.75)),
        "P10": float(np.quantile(a, 0.10)),
        "P90": float(np.quantile(a, 0.90)),
    }


def bootstrap_ci(values, statistic, rng, resamples=5000):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    observed = float(statistic(a))
    boot = np.empty(resamples, dtype=float)
    for i in range(resamples):
        boot[i] = statistic(a[rng.integers(0, len(a), size=len(a))])
    return observed, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def add_violin_box(ax, arrays, labels, ylabel, zero=False):
    parts = ax.violinplot(arrays, showmeans=False, showmedians=False, showextrema=False)
    colors = ["#3B82F6", "#10B981", "#EF4444"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.45)
    ax.boxplot(arrays, widths=0.18, showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel(ylabel)
    if zero:
        ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.grid(axis="y", alpha=0.2)


def summarize():
    HERE.mkdir(parents=True, exist_ok=True)
    frames = {}
    scalars = []
    for key, spec in ANALYSES.items():
        frame = pd.read_csv(HERE / f"sample_metrics_{key}.csv")
        frame["task_key"] = key
        frame["task_label"] = spec["label"]
        frames[key] = frame
        scalars.append(json.loads((HERE / f"task_scalars_{key}.json").read_text()))

    summary_rows = []
    for key, frame in frames.items():
        for metric in SAMPLE_METRICS:
            summary_rows.append({"task": ANALYSES[key]["label"], "metric": metric, **describe(frame[metric])})
    pd.DataFrame(summary_rows).to_csv(HERE / "task_summary.csv", index=False)
    pd.DataFrame(scalars).to_csv(HERE / "task_level_scalars.csv", index=False)

    rng = np.random.default_rng(20261005)
    boot_rows = []
    boot_specs = {
        "median_normalized_error_gain": ("normalized_error_gain", np.median),
        "median_helpful_alignment": ("helpful_alignment", np.median),
        "median_downstream_sensitivity": ("downstream_sensitivity", np.median),
        "fraction_improved": ("error_gain", lambda a: np.mean(a > 1e-10)),
    }
    for key, frame in frames.items():
        for name, (column, statistic) in boot_specs.items():
            point, low, high = bootstrap_ci(frame[column], statistic, rng)
            boot_rows.append(
                {"task": ANALYSES[key]["label"], "metric": name, "estimate": point, "ci95_low": low, "ci95_high": high, "resamples": 5000, "bootstrap_seed": 20261005}
            )
    pd.DataFrame(boot_rows).to_csv(HERE / "bootstrap_summary.csv", index=False)

    grad_rows, dir_rows, mag_rows = [], [], []
    for key, frame in frames.items():
        label = ANALYSES[key]["label"]
        grad_rows.append(
            {
                "task": label,
                "N": len(frame),
                "pearson_first_order_vs_delta_y": safe_corr(frame["first_order_shift"], frame["delta_y"], "pearson"),
                "spearman_first_order_vs_delta_y": safe_corr(frame["first_order_shift"], frame["delta_y"], "spearman"),
            }
        )
        dir_rows.append(
            {"task": label, "N": len(frame), "spearman_helpful_alignment_vs_normalized_error_gain": safe_corr(frame["helpful_alignment"], frame["normalized_error_gain"], "spearman")}
        )
        for predictor in ("rho_raw", "projection_scale"):
            mag_rows.append(
                {"task": label, "predictor": predictor, "N": len(frame), "spearman_vs_normalized_error_gain": safe_corr(frame[predictor], frame["normalized_error_gain"], "spearman")}
            )
    pd.DataFrame(grad_rows).to_csv(HERE / "gradient_shift_correlations.csv", index=False)
    pd.DataFrame(dir_rows).to_csv(HERE / "direction_vs_gain_correlations.csv", index=False)
    pd.DataFrame(mag_rows).to_csv(HERE / "magnitude_vs_gain_correlations.csv", index=False)

    overview = []
    for key, frame in frames.items():
        row = dict(next(s for s in scalars if s["label"] == ANALYSES[key]["label"]))
        row.update(
            {
                "median_rho_raw": float(frame["rho_raw"].median()),
                "median_projection_scale": float(frame["projection_scale"].median()),
                "median_rho_final": float(frame["rho_final"].median()),
                "median_normalized_error_gain": float(frame["normalized_error_gain"].median()),
                "median_target_directed_shift": float(frame["target_directed_shift"].median()),
                "median_helpful_alignment": float(frame["helpful_alignment"].median()),
                "median_downstream_sensitivity": float(frame["downstream_sensitivity"].median()),
                "mean_clipped": float(frame["clipped"].mean()),
            }
        )
        overview.append(row)
    pd.DataFrame(overview).to_csv(HERE / "task_overview.csv", index=False)

    labels = [ANALYSES[k]["label"] for k in ANALYSES]
    ordered = [frames[k] for k in ANALYSES]
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    add_violin_box(ax, [f["normalized_error_gain"] for f in ordered], labels, "Normalized error gain", zero=True)
    fig.tight_layout(); fig.savefig(HERE / "fig_error_gain_distribution.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    positions, data, colors, ticks = [], [], [], []
    pos = 1
    for label, frame in zip(labels, ordered):
        helped = frame.loc[frame["error_gain"] > 1e-10, "helpful_alignment"].to_numpy()
        hurt = frame.loc[frame["error_gain"] < -1e-10, "helpful_alignment"].to_numpy()
        data.extend([helped, hurt]); positions.extend([pos - 0.18, pos + 0.18]); colors.extend(["#10B981", "#EF4444"]); ticks.append(pos); pos += 1
    bp = ax.boxplot(data, positions=positions, widths=0.28, showfliers=False, patch_artist=True)
    for box, color in zip(bp["boxes"], colors): box.set_facecolor(color); box.set_alpha(0.55)
    ax.set_xticks(ticks, labels); ax.axhline(0, color="#555", ls="--", lw=1); ax.set_ylabel("Helpful alignment")
    ax.plot([], [], color="#10B981", lw=8, alpha=.55, label="error gain > 0"); ax.plot([], [], color="#EF4444", lw=8, alpha=.55, label="error gain < 0")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(HERE / "fig_helpful_alignment.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for ax, label, frame in zip(axes, labels, ordered):
        ax.scatter(frame["helpful_alignment"], frame["normalized_error_gain"], s=8, alpha=.35)
        rho = safe_corr(frame["helpful_alignment"], frame["normalized_error_gain"], "spearman")
        ax.axhline(0, color="#555", ls="--", lw=.8); ax.axvline(0, color="#555", ls=":", lw=.8)
        ax.set_title(f"{label}\nSpearman={rho:.3f}"); ax.set_xlabel("Helpful alignment"); ax.grid(alpha=.15)
    axes[0].set_ylabel("Normalized error gain"); fig.tight_layout(); fig.savefig(HERE / "fig_alignment_vs_gain.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for ax, label, frame in zip(axes, labels, ordered):
        x, y = frame["first_order_shift"], frame["delta_y"]
        ax.scatter(x, y, s=8, alpha=.35)
        lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color="#EF4444", lw=1, label="y=x")
        pr = safe_corr(x, y, "pearson"); sr = safe_corr(x, y, "spearman")
        ax.set_title(f"{label}\nPearson={pr:.3f}, Spearman={sr:.3f}"); ax.set_xlabel("First-order shift"); ax.grid(alpha=.15)
    axes[0].set_ylabel("Actual prediction shift"); fig.tight_layout(); fig.savefig(HERE / "fig_first_order_vs_actual_shift.pdf"); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.8), sharey="row")
    for col, (label, frame) in enumerate(zip(labels, ordered)):
        for row, predictor in enumerate(("rho_raw", "projection_scale")):
            ax = axes[row, col]
            ax.scatter(frame[predictor], frame["normalized_error_gain"], s=8, alpha=.35)
            rho = safe_corr(frame[predictor], frame["normalized_error_gain"], "spearman")
            ax.axhline(0, color="#555", ls="--", lw=.8); ax.set_title(f"{label}: Spearman={rho:.3f}"); ax.set_xlabel(predictor); ax.grid(alpha=.15)
    axes[0, 0].set_ylabel("Normalized error gain"); axes[1, 0].set_ylabel("Normalized error gain")
    fig.tight_layout(); fig.savefig(HERE / "fig_magnitude_vs_gain.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    add_violin_box(ax, [f["downstream_sensitivity"] for f in ordered], labels, "Downstream sensitivity", zero=False)
    ax.set_yscale("log"); fig.tight_layout(); fig.savefig(HERE / "fig_downstream_sensitivity.pdf"); plt.close(fig)

    report = """# CartNet GPSR post-hoc residual-direction mechanism analysis\n\nResults were generated from the fixed seed-42 minimum-validation-MAE CartNet-GPSR checkpoints. No training, checkpoint reselection, or hyperparameter change was performed. The residual-OFF predictions are within-checkpoint mechanistic counterfactuals and are not the independently trained CartNet baselines. See `task_overview.csv`, `task_summary.csv`, and the correlation tables for the computed results.\n"""
    (HERE / "posthoc_mechanism_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"event": "posthoc_summary_complete", "tasks": labels}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=ANALYSES)
    group.add_argument("--summarize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        run_task(args.task, args.batch_size)


if __name__ == "__main__":
    main()
