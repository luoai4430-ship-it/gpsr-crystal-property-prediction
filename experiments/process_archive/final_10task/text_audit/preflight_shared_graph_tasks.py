from __future__ import annotations

import argparse
import json
import math
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from run import configure
from data.dataset import get_dataloader


DATA = "/root/autodl-tmp/dataset1/dataset"
TASKS = {
    "Jarvis-Bandgap_OPT": "jarvis/optb88vdw_bandgap",
    "Jarvis-TotalEnergy": "jarvis/optb88vdw_total_energy",
    "Jarvis-ShearModulusGv": "jarvis/shear_modulus_gv",
    "MP-BulkModuli": "mp_2018_small/bulk",
}
EXPECTED = {
    "Jarvis-Bandgap_OPT": (44578, 5572, 5573),
    "Jarvis-TotalEnergy": (44578, 5572, 5573),
    "Jarvis-ShearModulusGv": (15744, 1968, 1968),
    "MP-BulkModuli": (4664, 583, 583),
}


def main() -> None:
    report = {}
    for task, rel in TASKS.items():
        args = argparse.Namespace(
            task="test",
            target=task,
            model="graphonly",
            use_cif_csv=True,
            cif_dir=os.path.join(DATA, rel, "cif"),
            text_emb_path=os.path.join(DATA, rel, "MatBert_embeddings.pt"),
            num_epoch=60,
            batch_size=32,
            seed=42,
            lr=1e-3,
            max_lr=1e-3,
            weight_decay=1e-4,
            alpha_init=None,
            alpha_max=0.2,
            lambda_delta=0.0,
            max_residual_ratio=float("inf"),
            lambda_res=0.0,
            rho_target=0.10,
            sample_gate_bias=3.0,
            inject_position="early",
            inject_after=0,
            text_shuffle_seed=-1,
            share_text_cache=True,
        )
        configure(args)
        train, valid, test, mean, std = get_dataloader()
        counts = (len(train.dataset), len(valid.dataset), len(test.dataset))
        if counts != EXPECTED[task]:
            raise RuntimeError(f"Unexpected split for {task}: {counts} != {EXPECTED[task]}")
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise RuntimeError(f"Invalid target normalization for {task}: mean={mean}, std={std}")
        first = next(iter(test))
        if len(first.cid) != 1 or first.text_emb.numel() != 768:
            raise RuntimeError(f"Malformed shared graph batch for {task}")
        report[task] = {
            "split_counts": counts,
            "target_mean_train": mean,
            "target_std_train": std,
            "first_test_id": str(first.cid[0]),
            "first_test_text_norm": float(first.text_emb.reshape(-1).norm()),
        }
        del train, valid, test, first
    output = os.path.join(ROOT, "experiments", "final_10task", "text_audit", "shared_graph_preflight.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"status": "PASS", "tasks": report}, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
