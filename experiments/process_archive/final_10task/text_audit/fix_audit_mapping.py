from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import zipfile
from collections import Counter
from pathlib import Path

import torch

from prepare_jarvis_10task import ARCHIVE, DATA_ROOT, REPO, embed_missing, embedding_tensor, valid_value, write_csv, write_id_prop


SOURCE_TASKS = {
    "mbj_bandgap": ("mbj_bandgap", "Bandgap_MBJ"),
    "formation_energy_peratom": ("formation_energy_peratom", "FormationEnergy"),
    "bulk_modulus_kv": ("bulk_modulus_kv", "BulkModulusKv"),
}
ALIASES = {
    "optb88vdw_bandgap": ("optb88vdw_bandgap", "formation_energy_peratom"),
    "optb88vdw_total_energy": ("optb88vdw_total_energy", "formation_energy_peratom"),
    "shear_modulus_gv": ("shear_modulus_gv", "bulk_modulus_kv"),
}


def load_records() -> list[dict]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        return json.load(archive.open([name for name in archive.namelist() if name.endswith(".json")][0]))


def stable_ids(records: list[dict]) -> dict[int, str]:
    jid_counts = Counter(str(record["jid"]) for record in records)
    result = {}
    used = set()
    duplicate_ordinals = Counter()
    for index, record in enumerate(records):
        jid = str(record["jid"])
        if jid_counts[jid] == 1:
            sid = jid
        else:
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            fingerprint = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
            sid = f"{jid}__record_{fingerprint}"
            if sid in used:
                duplicate_ordinals[sid] += 1
                sid = f"{sid}__copy_{duplicate_ordinals[sid] + 1}"
        if sid in used:
            raise RuntimeError(f"Stable record identifier collision: {sid}")
        used.add(sid)
        result[id(record)] = sid
    return result


def patch_source(task_dir_name: str, prop: str, target: str, records: list[dict], sid_map: dict[int, str]) -> dict:
    task_dir = DATA_ROOT / task_dir_name
    filtered = [record for record in records if valid_value(record.get(prop))]
    rows = list(csv.DictReader((task_dir / "description.csv").open(encoding="utf-8")))
    if len(rows) != len(filtered):
        raise RuntimeError(f"Description/order mismatch for {task_dir_name}")
    old_embeddings = torch.load(task_dir / "MatBert_embeddings.pt", map_location="cpu", weights_only=False)
    duplicate_positions = [index for index, record in enumerate(filtered) if "__record_" in sid_map[id(record)]]
    duplicate_vectors = embed_missing([rows[index]["Description"] for index in duplicate_positions])
    duplicate_by_pos = {position: duplicate_vectors[i] for i, position in enumerate(duplicate_positions)}
    payload = {}
    mapping_rows = []
    for index, (record, row) in enumerate(zip(filtered, rows)):
        old_jid = str(record["jid"])
        sid = sid_map[id(record)]
        if index in duplicate_by_pos:
            vec = duplicate_by_pos[index]
        else:
            vec = embedding_tensor(old_embeddings[old_jid])
        payload[sid] = {"embedding": vec.float().reshape(-1)}
        row["Id"] = sid
        row["material_id"] = sid
        row["source_index"] = index
        mapping_rows.append({"source_index": index, "jid": old_jid, "stable_record_id": sid})
    write_csv(task_dir / "description.csv", rows, list(rows[0]))
    write_csv(task_dir / "stable_id_map.csv", mapping_rows, ["source_index", "jid", "stable_record_id"])
    write_id_prop(task_dir / "cif" / "id_prop.csv", [dict(record, jid=sid_map[id(record)]) for record in filtered], prop)
    tmp = task_dir / "MatBert_embeddings.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, task_dir / "MatBert_embeddings.pt")

    raw_path = REPO / "Dataset" / "raw" / f"Jarvis-{target}-cutoff4.0-maxNeibNum16-spheHarmL4-CIF_CSV-rawData.pt"
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    if len(raw) != len(filtered):
        raise RuntimeError(f"Raw cache length mismatch for {target}")
    ordered_ids = [sid_map[id(record)] for record in filtered]
    for item, sid in zip(raw, ordered_ids):
        item.cid = sid
        item.text_emb = payload[sid]["embedding"].clone()
    raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    torch.save(raw, raw_tmp)
    os.replace(raw_tmp, raw_path)
    del raw

    shuffled_ids = list(ordered_ids)
    random.Random(123).shuffle(shuffled_ids)
    n_train = int(0.8 * len(shuffled_ids))
    n_valid = int(0.1 * len(shuffled_ids))
    split_ids = {
        "train": shuffled_ids[:n_train],
        "valid": shuffled_ids[n_train : n_train + n_valid],
        "test": shuffled_ids[n_train + n_valid :],
    }
    pattern = (
        f"Jarvis-{target}-cutoff4.0-maxNeibNum16-spheHarmL4-splitSeed123-"
        "CIF_CSV-{split}-CGCNN-multimodal-MatBert_embeddings.pt"
    )
    for split, expected_ids in split_ids.items():
        path = REPO / "Dataset" / "processed" / pattern.format(split=split)
        data, slices = torch.load(path, map_location="cpu", weights_only=False)
        if len(data.cid) != len(expected_ids):
            raise RuntimeError(f"Processed split length mismatch {target}/{split}")
        data.cid = expected_ids
        data.text_emb = torch.stack([payload[sid]["embedding"] for sid in expected_ids]).reshape(-1)
        slices["text_emb"] = torch.arange(0, (len(expected_ids) + 1) * 768, 768)
        cache_tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save((data, slices), cache_tmp)
        os.replace(cache_tmp, path)
    return {
        "task": task_dir_name,
        "num_samples": len(filtered),
        "fingerprinted_duplicate_jid_records": len(duplicate_positions),
        "num_unique_stable_ids": len(payload),
    }


def patch_alias(task_dir_name: str, prop: str, source_name: str, records: list[dict], sid_map: dict[int, str]) -> dict:
    task_dir = DATA_ROOT / task_dir_name
    source_dir = DATA_ROOT / source_name
    filtered = [record for record in records if valid_value(record.get(prop))]
    source_rows = list(csv.DictReader((source_dir / "description.csv").open(encoding="utf-8")))
    if len(source_rows) != len(filtered):
        raise RuntimeError(f"Alias/source order mismatch for {task_dir_name}")
    rows = []
    for index, (record, source) in enumerate(zip(filtered, source_rows)):
        sid = sid_map[id(record)]
        rows.append(
            {
                "Id": sid,
                "material_id": sid,
                "Composition": record.get("formula", ""),
                "prop": record[prop],
                "Description": source["Description"],
                "File_Name": source["File_Name"],
                "source_index": index,
                "text_source": f"shared_from_{source_name}",
            }
        )
    write_csv(task_dir / "description.csv", rows, list(rows[0]))
    write_id_prop(task_dir / "cif" / "id_prop.csv", [dict(record, jid=sid_map[id(record)]) for record in filtered], prop)
    target_embedding = task_dir / "MatBert_embeddings.pt"
    target_embedding.unlink(missing_ok=True)
    os.link(source_dir / "MatBert_embeddings.pt", target_embedding)
    return {"task": task_dir_name, "num_samples": len(rows), "num_unique_stable_ids": len({r['Id'] for r in rows})}


def patch_mp_raw_text_tables() -> dict:
    root = Path("/root/autodl-tmp/dataset1/dataset/mp_2018")
    generated_path = root / "MP_text_repaired" / "generated_missing_descriptions.csv"
    generated = {
        row["id"]: row
        for row in csv.DictReader(generated_path.open(encoding="utf-8"))
    }
    outputs = {}
    for task_dir_name in ("MP-bandgap", "MP-formation"):
        task_dir = root / task_dir_name
        rows = list(csv.DictReader((task_dir / "description.csv").open(encoding="utf-8-sig")))
        by_id = {row["id"].strip(): row for row in rows}
        targets = {}
        with (task_dir / "cif" / "id_prop.csv").open(newline="", encoding="utf-8-sig") as handle:
            for item in csv.reader(handle):
                targets[item[0].strip()] = item[1]
        added = 0
        for sid, generated_row in generated.items():
            if sid in targets and sid not in by_id:
                rows.append(
                    {
                        "id": sid,
                        "composition": generated_row.get("composition", ""),
                        "prop": targets[sid],
                        "text": generated_row["text"],
                        "File_Name": "generated_from_structure",
                    }
                )
                added += 1
        write_csv(task_dir / "description.csv", rows, list(rows[0]))
        outputs[task_dir_name] = {"added_generated_descriptions": added, "total_description_rows": len(rows)}
    return outputs


def main() -> None:
    records = load_records()
    sid_map = stable_ids(records)
    report = {"jarvis_sources": [], "jarvis_aliases": [], "mp_large": patch_mp_raw_text_tables()}
    for task, (prop, target) in SOURCE_TASKS.items():
        report["jarvis_sources"].append(patch_source(task, prop, target, records, sid_map))
    for task, (prop, source) in ALIASES.items():
        report["jarvis_aliases"].append(patch_alias(task, prop, source, records, sid_map))
    output = REPO / "experiments" / "final_10task" / "text_audit" / "mapping_collision_repair.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
