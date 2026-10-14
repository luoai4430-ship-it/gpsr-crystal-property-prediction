from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


REPO = Path("/root/autodl-tmp/TextResidualSHFMat-FAIR")
OUT = REPO / "experiments" / "final_10task" / "text_audit"
MODEL_DIR = Path("/root/autodl-tmp/matbert-base-cased")

TASKS = {
    "Jarvis-Bandgap_MBJ": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/jarvis/mbj_bandgap"),
        "cache_target": "Jarvis-Bandgap_MBJ",
        "dataset": "jarvis",
    },
    "Jarvis-Bandgap_OPT": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/jarvis/optb88vdw_bandgap"),
        "cache_target": "Jarvis-FormationEnergy",
        "dataset": "jarvis",
    },
    "Jarvis-FormationEnergy": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/jarvis/formation_energy_peratom"),
        "cache_target": "Jarvis-FormationEnergy",
        "dataset": "jarvis",
    },
    "Jarvis-TotalEnergy": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/jarvis/optb88vdw_total_energy"),
        "cache_target": "Jarvis-FormationEnergy",
        "dataset": "jarvis",
    },
    "Jarvis-BulkModulusKv": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/jarvis/bulk_modulus_kv"),
        "cache_target": "Jarvis-BulkModulusKv",
        "dataset": "jarvis",
    },
    "Jarvis-ShearModulusGv": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/jarvis/shear_modulus_gv"),
        "cache_target": "Jarvis-BulkModulusKv",
        "dataset": "jarvis",
    },
    "MP-Bandgap": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/mp_2018/MP-bandgap"),
        "cache_target": "MP-Bandgap",
        "dataset": "mp",
    },
    "MP-FormationEnergy": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/mp_2018/MP-formation"),
        "cache_target": "MP-FormationEnergy",
        "dataset": "mp",
    },
    "MP-BulkModuli": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/mp_2018_small/bulk"),
        "cache_target": "MP-ShearModuli",
        "dataset": "mp",
    },
    "MP-ShearModuli": {
        "dir": Path("/root/autodl-tmp/dataset1/dataset/mp_2018_small/shear"),
        "cache_target": "MP-ShearModuli",
        "dataset": "mp",
    },
}


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def load_id_prop(path: Path) -> list[str]:
    ids = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            sample_id = row[0].strip()
            if sample_id.lower() in {"id", "material_id", "cif_id"}:
                continue
            ids.append(sample_id)
    return ids


def choose_column(fieldnames: list[str], candidates: list[str]) -> str:
    lower = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"None of {candidates} in {fieldnames}")


def load_descriptions(path: Path) -> tuple[dict[str, str], dict[str, dict], int]:
    by_id = {}
    metadata = {}
    collisions = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        id_col = choose_column(reader.fieldnames or [], ["material_id", "id", "Id", "cif_id"])
        text_col = choose_column(reader.fieldnames or [], ["Description", "text", "description"])
        for row in reader:
            sample_id = (row.get(id_col) or "").strip()
            text = normalize_text(row.get(text_col) or "")
            if sample_id in by_id and by_id[sample_id] != text:
                collisions += 1
            by_id[sample_id] = text
            metadata[sample_id] = row
    return by_id, metadata, collisions


def vector(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().reshape(-1)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float().reshape(-1)
    if isinstance(value, dict):
        for key in ("embedding", "vec", "features", "emb", "hidden"):
            if key in value:
                return vector(value[key])
    raise TypeError(type(value))


def embedding_metrics(embeddings: dict, ids: list[str], seed: int = 42) -> dict:
    count = len(ids)
    sums = torch.zeros(768, dtype=torch.float64)
    sums_sq = torch.zeros(768, dtype=torch.float64)
    norms = []
    unique_hashes = set()
    for start in range(0, count, 1024):
        chunk = torch.stack([vector(embeddings[sid]) for sid in ids[start : start + 1024]])
        if chunk.shape[1] != 768:
            raise RuntimeError(f"Expected embedding dimension 768, got {tuple(chunk.shape)}")
        if not torch.isfinite(chunk).all():
            raise RuntimeError("Non-finite embedding value")
        sums += chunk.double().sum(0)
        sums_sq += chunk.double().square().sum(0)
        norms.extend(chunk.norm(dim=1).tolist())
        unique_hashes.update(hashlib.sha1(row.numpy().tobytes()).hexdigest() for row in chunk)
    mean = sums / count
    var = (sums_sq / count - mean.square()).clamp_min(0)
    dim_std = var.sqrt().numpy()

    rng = random.Random(seed)
    pairs = []
    while len(pairs) < min(10_000, max(1, count * 2)):
        a = rng.randrange(count)
        b = rng.randrange(count - 1)
        if b >= a:
            b += 1
        pairs.append((a, b))
    needed = sorted(set(index for pair in pairs for index in pair))
    selected = {index: vector(embeddings[ids[index]]) for index in needed}
    left = torch.stack([selected[a] for a, _ in pairs])
    right = torch.stack([selected[b] for _, b in pairs])
    cosine = F.cosine_similarity(left, right, dim=1).numpy()
    return {
        "num_unique_embeddings": len(unique_hashes),
        "embedding_duplicate_ratio": 1.0 - len(unique_hashes) / count,
        "mean_embedding_norm": float(np.mean(norms)),
        "std_embedding_norm": float(np.std(norms)),
        "mean_dimension_std": float(dim_std.mean()),
        "median_dimension_std": float(np.median(dim_std)),
        "min_dimension_std": float(dim_std.min()),
        "max_dimension_std": float(dim_std.max()),
        "fraction_dimensions_std_lt_1e-8": float((dim_std < 1e-8).mean()),
        "fraction_dimensions_std_lt_1e-6": float((dim_std < 1e-6).mean()),
        "fraction_dimensions_std_lt_1e-4": float((dim_std < 1e-4).mean()),
        "mean_pairwise_cosine": float(cosine.mean()),
        "std_pairwise_cosine": float(cosine.std()),
        "median_pairwise_cosine": float(np.median(cosine)),
        "p5_pairwise_cosine": float(np.percentile(cosine, 5)),
        "p25_pairwise_cosine": float(np.percentile(cosine, 25)),
        "p75_pairwise_cosine": float(np.percentile(cosine, 75)),
        "p95_pairwise_cosine": float(np.percentile(cosine, 95)),
        "max_pairwise_cosine": float(cosine.max()),
    }


def processed_paths(cache_target: str) -> list[Path]:
    prefix = (
        f"{cache_target}-cutoff4.0-maxNeibNum16-spheHarmL4-splitSeed123-"
        "CIF_CSV-{split}-CGCNN-multimodal-MatBert_embeddings.pt"
    )
    return [REPO / "Dataset" / "processed" / prefix.format(split=split) for split in ("train", "valid", "test")]


def load_training_cache_ids(cache_target: str) -> tuple[list[str], list[str]]:
    all_ids = []
    paths = processed_paths(cache_target)
    for path in paths:
        try:
            data, _ = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
        except TypeError:
            data, _ = torch.load(path, map_location="cpu", weights_only=False)
        all_ids.extend(str(value) for value in data.cid)
        del data
        gc.collect()
    return all_ids, [str(path) for path in paths]


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def audit_task(task: str, spec: dict, tokenizer) -> tuple[dict, list[dict], list[dict]]:
    task_dir = spec["dir"]
    ids = load_id_prop(task_dir / "cif" / "id_prop.csv")
    descriptions, description_rows, text_collisions = load_descriptions(task_dir / "description.csv")
    embeddings = torch.load(task_dir / "MatBert_embeddings.pt", map_location="cpu", weights_only=False)
    embedding_keys = {str(key) for key in embeddings}
    unique_ids = set(ids)
    missing_text = [sid for sid in ids if sid not in descriptions]
    missing_embedding = [sid for sid in ids if sid not in embedding_keys]
    extra_cache_keys = sorted(embedding_keys - unique_ids)
    empty_ids = [sid for sid in ids if not descriptions.get(sid, "")]
    fallback_phrases = ("default text", "missing description", "no description available", "unknown material")
    fallback_ids = [
        sid for sid in ids
        if any(phrase in descriptions.get(sid, "").lower() for phrase in fallback_phrases)
    ]
    reconstructed = sum(
        1
        for sid in ids
        if "reconstructed_structure_description" in description_rows.get(sid, {}).get("text_source", "")
    )
    text_hashes = {
        hashlib.sha256(descriptions.get(sid, "").encode("utf-8")).hexdigest()
        for sid in ids if descriptions.get(sid, "")
    }
    lengths = [len(descriptions.get(sid, "")) for sid in ids]
    metric_ids = [sid for sid in ids if sid in embeddings]
    emb_metrics = embedding_metrics(embeddings, metric_ids)

    cached_ids, cache_paths = load_training_cache_ids(spec["cache_target"])
    cached_set = set(cached_ids)
    cache_missing = sorted(unique_ids - cached_set)
    cache_extra = sorted(cached_set - unique_ids)

    rng = random.Random(12345 + list(TASKS).index(task))
    sample_ids = rng.sample(ids, min(20, len(ids)))
    chain = []
    for dataset_index, sid in [(ids.index(sid), sid) for sid in sample_ids]:
        vec = vector(embeddings[sid])
        meta = description_rows.get(sid, {})
        chain.append(
            {
                "task": task,
                "dataset_index": dataset_index,
                "sample_id": sid,
                "material_id_or_cif_id": meta.get("material_id") or meta.get("Id") or meta.get("id") or sid,
                "raw_text_first_300": descriptions[sid][:300],
                "cache_key": sid,
                "embedding_first_10": [float(x) for x in vec[:10]],
                "embedding_norm": float(vec.norm()),
                "training_cache_contains_id": sid in cached_set,
            }
        )

    token_ids = rng.sample(ids, min(10, len(ids)))
    token_batch = tokenizer(
        [descriptions[sid] for sid in token_ids],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    token_rows = []
    token_sequences = set()
    for index, sid in enumerate(token_ids):
        input_ids = token_batch["input_ids"][index].tolist()
        attention = token_batch["attention_mask"][index].tolist()
        attended_ids = tuple(token for token, mask in zip(input_ids, attention) if mask)
        token_sequences.add(attended_ids)
        token_rows.append(
            {
                "task": task,
                "sample_id": sid,
                "input_ids": input_ids,
                "attention_mask": attention,
                "decoded": tokenizer.decode(input_ids, skip_special_tokens=False),
            }
        )

    num_key_collisions = len(ids) - len(unique_ids) + text_collisions
    status = "PASS"
    if num_key_collisions:
        status = "CACHE_KEY_COLLISION"
    elif missing_text or missing_embedding or cache_missing or cache_extra:
        status = "TEXT_MAPPING_ERROR"
    elif empty_ids or fallback_ids or len(text_hashes) / len(ids) < 0.90:
        status = "TEXT_DUPLICATION"
    elif (
        emb_metrics["mean_pairwise_cosine"] > 0.999
        and emb_metrics["std_pairwise_cosine"] < 1e-3
    ) or emb_metrics["num_unique_embeddings"] / len(ids) < 0.90:
        status = "EMBEDDING_COLLAPSE"
    elif len(token_sequences) < 2:
        status = "TOKENIZATION_ERROR"

    result = {
        "task": task,
        "num_samples": len(ids),
        "num_unique_sample_ids": len(unique_ids),
        "num_unique_ids": len(unique_ids),
        "num_unique_raw_texts": len(text_hashes),
        "num_unique_texts": len(text_hashes),
        "unique_text_ratio": len(text_hashes) / len(ids),
        "text_duplicate_ratio": 1.0 - len(text_hashes) / len(ids),
        "num_unique_cache_keys": len(embedding_keys & unique_ids),
        "num_key_collisions": num_key_collisions,
        "num_extra_cache_keys": len(extra_cache_keys),
        "mean_text_length": float(np.mean(lengths)),
        "median_text_length": float(np.median(lengths)),
        "p10_text_length": percentile(lengths, 10),
        "p90_text_length": percentile(lengths, 90),
        "fallback_count": len(fallback_ids),
        "empty_text_count": len(empty_ids),
        "reconstructed_from_structure_count": reconstructed,
        "missing_text_count": len(missing_text),
        "missing_embedding_count": len(missing_embedding),
        "training_cache_missing_id_count": len(cache_missing),
        "training_cache_extra_id_count": len(cache_extra),
        "tokenizer_unique_sequences_in_10": len(token_sequences),
        "attention_mask_forward_required": True,
        "cache_files": " | ".join(cache_paths),
        **emb_metrics,
        "audit_status": status,
    }
    del embeddings
    gc.collect()
    return result, chain, token_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def old_new_mp_comparison() -> None:
    rows = []
    for task, directory in (
        ("MP-Bandgap", TASKS["MP-Bandgap"]["dir"]),
        ("MP-FormationEnergy", TASKS["MP-FormationEnergy"]["dir"]),
    ):
        ids = load_id_prop(directory / "cif" / "id_prop.csv")
        current = torch.load(directory / "MatBert_embeddings.pt", map_location="cpu", weights_only=False)
        backup_candidates = sorted(directory.glob("MatBert_embeddings.pt.degenerate_*.pt"))
        if not backup_candidates:
            continue
        old = torch.load(backup_candidates[-1], map_location="cpu", weights_only=False)
        for state, payload in (("before", old), ("after", current)):
            metric_ids = [sid for sid in ids if sid in payload]
            metrics = embedding_metrics(payload, metric_ids, seed=2026)
            rows.append(
                {
                    "task": task,
                    "state": state,
                    "num_samples": len(metric_ids),
                    "num_unique_embeddings": metrics["num_unique_embeddings"],
                    "mean_pairwise_cosine": metrics["mean_pairwise_cosine"],
                    "std_pairwise_cosine": metrics["std_pairwise_cosine"],
                    "mean_dimension_std": metrics["mean_dimension_std"],
                    "mean_embedding_norm": metrics["mean_embedding_norm"],
                }
            )
        del current, old
        gc.collect()
    write_csv(OUT / "mp_embedding_before_after.csv", rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    results = []
    chains = []
    token_rows = []
    for task, spec in TASKS.items():
        print(f"Auditing {task}...", flush=True)
        try:
            result, chain, token = audit_task(task, spec, tokenizer)
        except Exception as error:
            result = {"task": task, "audit_status": "UNKNOWN_ERROR", "error": repr(error)}
            chain, token = [], []
        results.append(result)
        chains.extend(chain)
        token_rows.extend(token)
        print(task, result.get("audit_status"), flush=True)
    fieldnames = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (OUT / "text_audit_10task.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    (OUT / "sample_mapping_chains.json").write_text(json.dumps(chains, indent=2), encoding="utf-8")
    (OUT / "tokenizer_audit.json").write_text(json.dumps(token_rows, indent=2), encoding="utf-8")
    code_audit = """# MatBERT batch encoding audit\n\nThe active embedding implementation tokenizes a batch to `[B,L]`, passes the full tokenizer dictionary including `attention_mask` into `model(**inputs)`, and extracts `outputs.last_hidden_state[:, 0, :]`, producing `[B,768]`. No batch-first-item replication, `repeat`, `expand`, or batch-axis mean is used. `audit_10task.py` additionally records ten tokenized examples per task and verifies more than one unique attended token sequence.\n"""
    (OUT / "batch_encoding_code_audit.md").write_text(code_audit, encoding="utf-8")
    old_new_mp_comparison()
    all_pass = len(results) == 10 and all(row.get("audit_status") == "PASS" for row in results)
    completion = {
        "stage": "text_audit",
        "all_pass": all_pass,
        "passed": sum(row.get("audit_status") == "PASS" for row in results),
        "total": len(results),
        "statuses": {row["task"]: row.get("audit_status") for row in results},
    }
    (OUT / "completion_status.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
