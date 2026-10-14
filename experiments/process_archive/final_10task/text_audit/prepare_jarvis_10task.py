from __future__ import annotations

import csv
import json
import math
import os
import shutil
import zipfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


REPO = Path("/root/autodl-tmp/TextResidualSHFMat-FAIR")
DATA_ROOT = Path("/root/autodl-tmp/dataset1/dataset/jarvis")
ARCHIVE = Path("/root/autodl-tmp/dataset1/jdft_3d-8-18-2021.json.zip")
MODEL_DIR = Path("/root/autodl-tmp/matbert-base-cased")

SOURCE_TASKS = {
    "mbj_bandgap": {
        "prop": "mbj_bandgap",
        "target": "Bandgap_MBJ",
    },
    "formation_energy_peratom": {
        "prop": "formation_energy_peratom",
        "target": "FormationEnergy",
    },
    "bulk_modulus_kv": {
        "prop": "bulk_modulus_kv",
        "target": "BulkModulusKv",
    },
}

ALIAS_TASKS = {
    "optb88vdw_bandgap": {
        "prop": "optb88vdw_bandgap",
        "source": "formation_energy_peratom",
    },
    "optb88vdw_total_energy": {
        "prop": "optb88vdw_total_energy",
        "source": "formation_energy_peratom",
    },
    "shear_modulus_gv": {
        "prop": "shear_modulus_gv",
        "source": "bulk_modulus_kv",
    },
}


def valid_value(value) -> bool:
    if value is None or value == "na" or value == "":
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def load_records() -> list[dict]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one JSON member in {ARCHIVE}, got {names}")
        return json.load(archive.open(names[0]))


def fallback_description(record: dict) -> str:
    atoms = record.get("atoms") or {}
    lattice = atoms.get("lattice_mat") or []
    coords = atoms.get("coords") or []
    elements = atoms.get("elements") or []
    lattice_text = "; ".join(
        "[" + ", ".join(f"{float(x):.5f}" for x in row) + "]"
        for row in lattice
    )
    sites = "; ".join(
        f"{element}({', '.join(f'{float(x):.5f}' for x in coord)})"
        for element, coord in list(zip(elements, coords))[:32]
    )
    return (
        f"A crystal with composition {record.get('formula', 'unknown')} has "
        f"{record.get('nat', len(elements))} atoms in its cell. The crystal system is "
        f"{record.get('crys', 'unknown')} with space group {record.get('spg_symbol', record.get('spg', 'unknown'))} "
        f"(number {record.get('spg_number', 'unknown')}) and dimensionality "
        f"{record.get('dimensionality', 'unknown')}. The lattice vectors in angstrom are "
        f"{lattice_text}. Fractional atomic sites are {sites}."
    )


def embedding_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().reshape(-1)
    if isinstance(value, dict):
        for key in ("embedding", "vec", "features", "emb", "hidden"):
            if isinstance(value.get(key), torch.Tensor):
                return value[key].detach().cpu().float().reshape(-1)
    raise TypeError(f"Cannot extract embedding from {type(value)}")


def embed_missing(texts: list[str]) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 768), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config = AutoConfig.from_pretrained(MODEL_DIR)
    kwargs = {"add_pooling_layer": False} if config.model_type == "bert" else {}
    model = AutoModel.from_pretrained(MODEL_DIR, **kwargs).to(device).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch = tokenizer(
                texts[start : start + 16],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output.append(model(**batch).last_hidden_state[:, 0, :].cpu())
    return torch.cat(output, dim=0)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_id_prop(path: Path, filtered: list[dict], prop: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for record in filtered:
            writer.writerow([record["jid"], record[prop]])
    os.replace(tmp, path)


def prepare_source(task_dir_name: str, spec: dict, records: list[dict]) -> dict:
    task_dir = DATA_ROOT / task_dir_name
    prop = spec["prop"]
    filtered = [record for record in records if valid_value(record.get(prop))]
    old_rows = list(csv.DictReader((task_dir / "description.csv").open(encoding="utf-8")))
    old_by_index = {int(row["Id"]): row for row in old_rows}
    old_embeddings = torch.load(task_dir / "MatBert_embeddings.pt", map_location="cpu", weights_only=False)

    missing_indices = [index for index in range(len(filtered)) if index not in old_by_index]
    missing_texts = [fallback_description(filtered[index]) for index in missing_indices]
    missing_vectors = embed_missing(missing_texts)
    missing_vector_by_index = {
        index: missing_vectors[pos] for pos, index in enumerate(missing_indices)
    }

    rows = []
    payload = {}
    mapping_rows = []
    for index, record in enumerate(filtered):
        jid = str(record["jid"])
        old = old_by_index.get(index)
        description = (old.get("Description") or "").strip() if old else fallback_description(record)
        if not description:
            description = fallback_description(record)
        if old is not None and str(index) in old_embeddings:
            vector = embedding_tensor(old_embeddings[str(index)])
            text_source = "existing_robo_description"
        else:
            vector = missing_vector_by_index[index].float().reshape(-1)
            text_source = "reconstructed_structure_description"
        if vector.numel() != 768 or not torch.isfinite(vector).all():
            raise RuntimeError(f"Bad embedding for {task_dir_name} {jid}: {tuple(vector.shape)}")
        payload[jid] = {"embedding": vector}
        rows.append(
            {
                "Id": jid,
                "material_id": jid,
                "Composition": record.get("formula", ""),
                "prop": record[prop],
                "Description": description,
                "File_Name": old.get("File_Name", "generated_from_structure") if old else "generated_from_structure",
                "source_index": index,
                "text_source": text_source,
            }
        )
        mapping_rows.append({"source_index": index, "material_id": jid})

    write_csv(
        task_dir / "description.csv",
        rows,
        ["Id", "material_id", "Composition", "prop", "Description", "File_Name", "source_index", "text_source"],
    )
    write_csv(task_dir / "stable_id_map.csv", mapping_rows, ["source_index", "material_id"])
    write_id_prop(task_dir / "cif" / "id_prop.csv", filtered, prop)
    emb_tmp = task_dir / "MatBert_embeddings.pt.tmp"
    torch.save(payload, emb_tmp)
    os.replace(emb_tmp, task_dir / "MatBert_embeddings.pt")
    return {
        "task_dir": task_dir_name,
        "prop": prop,
        "target": spec["target"],
        "filtered": filtered,
        "num_rows": len(rows),
        "num_missing_reconstructed": len(missing_indices),
        "embedding_path": str(task_dir / "MatBert_embeddings.pt"),
    }


def patch_source_caches(prepared: dict) -> None:
    target = prepared["target"]
    filtered = prepared["filtered"]
    embedding_path = Path(prepared["embedding_path"])
    embeddings = torch.load(embedding_path, map_location="cpu", weights_only=False)
    jid_by_index = [str(record["jid"]) for record in filtered]

    raw_path = REPO / "Dataset" / "raw" / (
        f"Jarvis-{target}-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt"
    )
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    if len(raw) != len(filtered):
        raise RuntimeError(f"Raw length mismatch for {target}: {len(raw)} != {len(filtered)}")
    for item in raw:
        old_index = int(item.cid)
        jid = jid_by_index[old_index]
        item.cid = jid
        item.text_emb = embeddings[jid]["embedding"].clone()
    tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    torch.save(raw, tmp)
    os.replace(tmp, raw_path)
    del raw

    pattern = (
        f"Jarvis-{target}-cutoff4.0-maxNeibNum16-spheHarmL4-splitSeed123-"
        "CIF_CSV-{split}-CGCNN-multimodal-MatBert_embeddings.pt"
    )
    for split in ("train", "valid", "test"):
        path = REPO / "Dataset" / "processed" / pattern.format(split=split)
        data, slices = torch.load(path, map_location="cpu", weights_only=False)
        old_ids = list(data.cid)
        new_ids = [jid_by_index[int(value)] for value in old_ids]
        data.cid = new_ids
        vectors = torch.stack([embeddings[jid]["embedding"] for jid in new_ids])
        data.text_emb = vectors.reshape(-1)
        if "text_emb" in slices:
            slices["text_emb"] = torch.arange(0, (len(new_ids) + 1) * 768, 768)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save((data, slices), tmp)
        os.replace(tmp, path)


def prepare_alias(task_dir_name: str, spec: dict, records: list[dict]) -> dict:
    source_dir = DATA_ROOT / spec["source"]
    task_dir = DATA_ROOT / task_dir_name
    task_dir.mkdir(parents=True, exist_ok=True)
    filtered = [record for record in records if valid_value(record.get(spec["prop"]))]
    source_rows = {
        row["material_id"]: row
        for row in csv.DictReader((source_dir / "description.csv").open(encoding="utf-8"))
    }
    rows = []
    for index, record in enumerate(filtered):
        jid = str(record["jid"])
        source = source_rows.get(jid)
        if source is None:
            raise RuntimeError(f"Alias {task_dir_name} missing source text for {jid}")
        rows.append(
            {
                "Id": jid,
                "material_id": jid,
                "Composition": record.get("formula", ""),
                "prop": record[spec["prop"]],
                "Description": source["Description"],
                "File_Name": source["File_Name"],
                "source_index": index,
                "text_source": f"shared_from_{spec['source']}",
            }
        )
    write_csv(
        task_dir / "description.csv",
        rows,
        ["Id", "material_id", "Composition", "prop", "Description", "File_Name", "source_index", "text_source"],
    )
    write_id_prop(task_dir / "cif" / "id_prop.csv", filtered, spec["prop"])
    target_emb = task_dir / "MatBert_embeddings.pt"
    if target_emb.exists() or target_emb.is_symlink():
        target_emb.unlink()
    os.link(source_dir / "MatBert_embeddings.pt", target_emb)
    return {
        "task_dir": task_dir_name,
        "prop": spec["prop"],
        "source": spec["source"],
        "num_rows": len(rows),
        "embedding_hardlink": os.stat(target_emb).st_ino == os.stat(source_dir / "MatBert_embeddings.pt").st_ino,
    }


def main() -> None:
    records = load_records()
    if len(records) != 55723:
        raise RuntimeError(f"Unexpected JARVIS 2021 record count: {len(records)}")
    manifest = {"archive": str(ARCHIVE), "num_official_records": len(records), "sources": [], "aliases": []}
    prepared_sources = []
    for task_dir_name, spec in SOURCE_TASKS.items():
        prepared = prepare_source(task_dir_name, spec, records)
        prepared_sources.append(prepared)
        manifest["sources"].append({key: value for key, value in prepared.items() if key != "filtered"})
    for prepared in prepared_sources:
        patch_source_caches(prepared)
    for task_dir_name, spec in ALIAS_TASKS.items():
        manifest["aliases"].append(prepare_alias(task_dir_name, spec, records))
    out = REPO / "experiments" / "final_10task" / "text_audit" / "jarvis_preparation_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
