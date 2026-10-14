from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import torch
from jarvis.core.atoms import Atoms
from transformers import AutoConfig, AutoModel, AutoTokenizer


REPO = Path("/root/autodl-tmp/TextResidualSHFMat-FAIR")
ROOT = Path("/root/autodl-tmp/dataset1/dataset/mp_2018_small")
MODEL_DIR = Path("/root/autodl-tmp/matbert-base-cased")


def stable_id(source_id: str | int) -> str:
    return f"mp-elastic-{int(source_id):06d}"


def extract_vector(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().reshape(-1)
    if isinstance(value, dict):
        for key in ("embedding", "vec", "features", "emb", "hidden"):
            if isinstance(value.get(key), torch.Tensor):
                return value[key].detach().cpu().float().reshape(-1)
    raise TypeError(type(value))


def describe_cif(path: Path) -> str:
    atoms = Atoms.from_cif(str(path), use_cif2cell=False)
    lattice = atoms.lattice_mat
    coords = atoms.frac_coords
    elements = atoms.elements
    lattice_text = "; ".join(
        "[" + ", ".join(f"{float(x):.5f}" for x in row) + "]"
        for row in lattice
    )
    sites = "; ".join(
        f"{element}({', '.join(f'{float(x):.5f}' for x in coord)})"
        for element, coord in list(zip(elements, coords))[:32]
    )
    composition = atoms.composition.reduced_formula
    return (
        f"A crystal with composition {composition} has {len(elements)} atoms in its cell. "
        f"The lattice vectors in angstrom are {lattice_text}. "
        f"Fractional atomic sites are {sites}."
    )


def embed(texts: list[str]) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config = AutoConfig.from_pretrained(MODEL_DIR)
    kwargs = {"add_pooling_layer": False} if config.model_type == "bert" else {}
    model = AutoModel.from_pretrained(MODEL_DIR, **kwargs).to(device).eval()
    chunks = []
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
            chunks.append(model(**batch).last_hidden_state[:, 0, :].cpu())
    return torch.cat(chunks, dim=0)


def read_targets(task: str) -> dict[str, float]:
    result = {}
    with (ROOT / task / "cif" / "id_prop.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            result[str(int(row[0]))] = float(row[1])
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    bulk_dir = ROOT / "bulk"
    source_rows = {
        str(int(row["Id"])): row
        for row in csv.DictReader((bulk_dir / "description.csv").open(encoding="utf-8"))
    }
    source_embeddings = torch.load(
        bulk_dir / "MatBert_embeddings.pt", map_location="cpu", weights_only=False
    )
    targets = {task: read_targets(task) for task in ("bulk", "shear")}
    expected_ids = sorted(targets["bulk"], key=int)
    if expected_ids != sorted(targets["shear"], key=int):
        raise RuntimeError("MP bulk/shear CIF identifiers differ")
    missing = [sample_id for sample_id in expected_ids if sample_id not in source_rows]
    missing_texts = [describe_cif(bulk_dir / "cif" / f"{sample_id}.cif") for sample_id in missing]
    missing_vectors = embed(missing_texts)
    missing_map = {
        sample_id: (text, missing_vectors[index])
        for index, (sample_id, text) in enumerate(zip(missing, missing_texts))
    }

    text_by_source_id = {}
    payload = {}
    for sample_id in expected_ids:
        if sample_id in source_rows:
            text = source_rows[sample_id]["Description"].strip()
            vector = extract_vector(source_embeddings[sample_id])
            text_source = "existing_robo_description"
        else:
            text, vector = missing_map[sample_id]
            text_source = "reconstructed_structure_description"
        sid = stable_id(sample_id)
        text_by_source_id[sample_id] = (text, text_source)
        payload[sid] = {"embedding": vector.float().reshape(-1)}

    canonical_embedding = bulk_dir / "MatBert_embeddings.pt"
    tmp = canonical_embedding.with_suffix(canonical_embedding.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, canonical_embedding)

    for task in ("bulk", "shear"):
        task_dir = ROOT / task
        rows = []
        for sample_id in expected_ids:
            sid = stable_id(sample_id)
            text, text_source = text_by_source_id[sample_id]
            rows.append(
                {
                    "Id": sid,
                    "material_id": sid,
                    "Composition": source_rows.get(sample_id, {}).get("Composition", ""),
                    "prop": targets[task][sample_id],
                    "Description": text,
                    "File_Name": source_rows.get(sample_id, {}).get("File_Name", "generated_from_structure"),
                    "source_cif_id": sample_id,
                    "text_source": text_source,
                }
            )
        write_csv(
            task_dir / "description.csv",
            rows,
            ["Id", "material_id", "Composition", "prop", "Description", "File_Name", "source_cif_id", "text_source"],
        )
        id_prop_tmp = task_dir / "cif" / "id_prop.csv.tmp"
        with id_prop_tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for sample_id in expected_ids:
                writer.writerow([stable_id(sample_id), targets[task][sample_id]])
                stable_cif = task_dir / "cif" / f"{stable_id(sample_id)}.cif"
                if not stable_cif.exists():
                    stable_cif.symlink_to(f"{sample_id}.cif")
        os.replace(id_prop_tmp, task_dir / "cif" / "id_prop.csv")
        target_embedding = task_dir / "MatBert_embeddings.pt"
        if task == "shear":
            if target_embedding.exists():
                target_embedding.unlink()
            os.link(canonical_embedding, target_embedding)

    shear_target = "ShearModuli"
    raw_path = REPO / "Dataset" / "raw" / (
        f"MP-{shear_target}-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt"
    )
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    if len(raw) != len(expected_ids):
        raise RuntimeError(f"Unexpected MP shear raw cache length: {len(raw)}")
    for data in raw:
        source_id = str(int(data.cid))
        sid = stable_id(source_id)
        data.cid = sid
        data.text_emb = payload[sid]["embedding"].clone()
    raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    torch.save(raw, raw_tmp)
    os.replace(raw_tmp, raw_path)

    processed_pattern = (
        f"MP-{shear_target}-cutoff4.0-maxNeibNum16-spheHarmL4-splitSeed123-"
        "CIF_CSV-{split}-CGCNN-multimodal-MatBert_embeddings.pt"
    )
    for split in ("train", "valid", "test"):
        path = REPO / "Dataset" / "processed" / processed_pattern.format(split=split)
        data, slices = torch.load(path, map_location="cpu", weights_only=False)
        new_ids = [stable_id(value) for value in data.cid]
        data.cid = new_ids
        data.text_emb = torch.stack([payload[sid]["embedding"] for sid in new_ids]).reshape(-1)
        slices["text_emb"] = torch.arange(0, (len(new_ids) + 1) * 768, 768)
        cache_tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save((data, slices), cache_tmp)
        os.replace(cache_tmp, path)

    manifest = {
        "status": "PASS",
        "num_samples": len(expected_ids),
        "missing_descriptions_reconstructed": len(missing),
        "missing_source_cif_ids": missing,
        "stable_id_format": "mp-elastic-%06d",
        "embedding_hardlinked_between_bulk_and_shear": (
            os.stat(ROOT / "bulk" / "MatBert_embeddings.pt").st_ino
            == os.stat(ROOT / "shear" / "MatBert_embeddings.pt").st_ino
        ),
    }
    output = REPO / "experiments" / "final_10task" / "text_audit" / "mp_elastic_repair_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
