"""Fixed benchmark records and shared official-CartNet geometry caches."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.specie import get_node_attributes
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data, InMemoryDataset

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/cartnet_gpsr"
DATA = Path(os.environ.get("GPSR_DATA_ROOT", "/root/autodl-tmp/dataset1/dataset"))
JARVIS_ARCHIVE = Path(os.environ.get("GPSR_JARVIS_ARCHIVE", "/root/autodl-tmp/dataset1/jdft_3d-8-18-2021.json.zip"))
CARTNET = ROOT / "third_party/CartNet"
sys.path.insert(0, str(CARTNET))
from dataset.utils import radius_graph_pbc


TASKS = {
    "Jarvis-Bandgap_MBJ": dict(rel="jarvis/mbj_bandgap", family="jarvis", group="j_mbj", log10=False, category="Electronic"),
    "Jarvis-Bandgap_OPT": dict(rel="jarvis/optb88vdw_bandgap", family="jarvis", group="j_energy", log10=False, category="Electronic"),
    "Jarvis-FormationEnergy": dict(rel="jarvis/formation_energy_peratom", family="jarvis", group="j_energy", log10=False, category="Energy"),
    "Jarvis-TotalEnergy": dict(rel="jarvis/optb88vdw_total_energy", family="jarvis", group="j_energy", log10=False, category="Energy"),
    "Jarvis-BulkModulusKv": dict(rel="jarvis/bulk_modulus_kv", family="jarvis", group="j_mech", log10=False, category="Mechanical"),
    "Jarvis-ShearModulusGv": dict(rel="jarvis/shear_modulus_gv", family="jarvis", group="j_mech", log10=False, category="Mechanical"),
    "MP-Bandgap": dict(rel="mp_2018/MP-bandgap", family="mp", group="mp_large", log10=False, category="Electronic"),
    "MP-FormationEnergy": dict(rel="mp_2018/MP-formation", family="mp", group="mp_large", log10=False, category="Energy"),
    "MP-BulkModuli": dict(rel="mp_2018_small/bulk", family="mp", group="mp_mech", log10=True, category="Mechanical"),
    "MP-ShearModuli": dict(rel="mp_2018_small/shear", family="mp", group="mp_mech", log10=True, category="Mechanical"),
}
GROUP_TASKS: Dict[str, List[str]] = {}
for _task, _spec in TASKS.items():
    GROUP_TASKS.setdefault(_spec["group"], []).append(_task)


@dataclass
class Record:
    cid: str
    y: float
    text: torch.Tensor


def _embedding_value(entry):
    if isinstance(entry, dict):
        entry = entry.get("embedding", entry.get("vec", entry.get("features")))
    return torch.as_tensor(entry, dtype=torch.float32).reshape(-1)


def load_records(task: str) -> List[Record]:
    spec = TASKS[task]
    root = DATA / spec["rel"]
    frame = pd.read_csv(root / "cif/id_prop.csv", header=None)
    emb = torch.load(root / "MatBert_embeddings.pt", map_location="cpu")
    records = []
    for cid, target in frame.iloc[:, :2].itertuples(index=False, name=None):
        cid = str(cid).strip()
        value = float(target)
        if spec["log10"]:
            value = math.log10(max(value, 1e-30))
        entry = emb.get(cid)
        if entry is None:
            try:
                entry = emb.get(int(cid))
            except ValueError:
                pass
        if entry is None:
            raise KeyError(f"missing text embedding for {task}/{cid}")
        text = _embedding_value(entry)
        if text.numel() != 768 or not torch.isfinite(text).all():
            raise ValueError(f"invalid text embedding for {task}/{cid}: {tuple(text.shape)}")
        records.append(Record(cid, value, text))
    return records


def fixed_split(task: str, records: List[Record]):
    records = list(records)
    random.Random(123).shuffle(records)
    n = len(records)
    if TASKS[task]["family"] == "jarvis" or n < 65000:
        n_train, n_valid = int(0.8 * n), int(0.1 * n)
    else:
        n_train, n_valid = 60000, 5000
    return records[:n_train], records[n_train:n_train+n_valid], records[n_train+n_valid:]


def write_and_check_manifest(task: str, splits, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ("train", "valid", "test")
    sets = [set(r.cid for r in s) for s in splits]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    expected = len(load_records(task))
    loaded = sum(map(len, splits))
    assert expected == loaded
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task", "split", "position", "material_id"))
        for name, split in zip(names, splits):
            for i, record in enumerate(split):
                writer.writerow((task, name, i, record.cid))
    return {"expected": expected, "loaded": loaded, "sizes": [len(s) for s in splits]}


def structure_id(cid: str) -> str:
    return cid.split("__record_", 1)[0]


class GeometryStore(InMemoryDataset):
    """One geometry cache per shared structure collection."""
    def __init__(self, group: str):
        self.group = group
        root = EXP / "geometry_cache" / group
        super().__init__(str(root))
        self.data, self.slices, self.cids = torch.load(self.processed_paths[0], map_location="cpu")
        self.by_cid = {cid: i for i, cid in enumerate(self.cids)}

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return f"{self.group}_cartnet_r5_all.pt"

    def download(self):
        pass

    def process(self):
        task_names = GROUP_TASKS[self.group]
        ids = []
        seen = set()
        for task in task_names:
            for record in load_records(task):
                if record.cid not in seen:
                    ids.append(record.cid); seen.add(record.cid)
        family = TASKS[task_names[0]]["family"]
        atoms_by_id = {}
        if family == "jarvis":
            archive = JARVIS_ARCHIVE
            with zipfile.ZipFile(archive) as handle:
                with handle.open(handle.namelist()[0]) as stream:
                    official = json.load(stream)
            wanted = {structure_id(cid) for cid in ids}
            base = {str(row["jid"]): row["atoms"] for row in official if str(row["jid"]) in wanted}
            missing = wanted - base.keys()
            if missing:
                raise RuntimeError(f"missing {len(missing)} JARVIS structures: {sorted(missing)[:10]}")
            atoms_by_id = {cid: base[structure_id(cid)] for cid in ids}
        else:
            cif_roots = [DATA / TASKS[t]["rel"] / "cif" for t in task_names]

        out = []
        progress = EXP / "logs" / f"geometry_{self.group}.jsonl"
        progress.parent.mkdir(parents=True, exist_ok=True)
        for i, cid in enumerate(ids):
            if family == "jarvis":
                atoms = Atoms.from_dict(atoms_by_id[cid])
            else:
                cif_path = next((p / f"{cid}.cif" for p in cif_roots if (p / f"{cid}.cif").is_file()), None)
                if cif_path is None:
                    raise FileNotFoundError(f"CIF missing for {self.group}/{cid}")
                atoms = Atoms.from_cif(str(cif_path), use_cif2cell=False)
            z = torch.tensor(
                [get_node_attributes(s, atom_features="atomic_number") for s in atoms.elements],
                dtype=torch.long,
            ).reshape(-1)
            data = Data(x=z)
            data.pos = torch.tensor(atoms.cart_coords, dtype=torch.float32)
            data.cell = torch.tensor(atoms.lattice.matrix, dtype=torch.float32).unsqueeze(0)
            data.pbc = torch.tensor([[True, True, True]])
            data.natoms = torch.tensor([z.numel()])
            one = Batch.from_data_list([data])
            # Official CartNet API: the third positional argument is the
            # neighbor cap.  None retains every neighbor within the radius.
            edge_index, _, _, vector = radius_graph_pbc(one, 5.0, None)
            data.edge_index = edge_index
            data.cart_dist = torch.linalg.vector_norm(vector, dim=-1)
            data.cart_dir = torch.nn.functional.normalize(vector, p=2, dim=-1)
            del data.pbc
            out.append(data)
            if (i + 1) % 500 == 0 or i + 1 == len(ids):
                event = {"group": self.group, "done": i + 1, "total": len(ids)}
                print(json.dumps(event), flush=True)
                with progress.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
        data, slices = self.collate(out)
        Path(self.processed_paths[0]).parent.mkdir(parents=True, exist_ok=True)
        torch.save((data, slices, ids), self.processed_paths[0])


class TaskDataset(Dataset):
    def __init__(self, records: List[Record], geometry: GeometryStore):
        self.records = records
        self.geometry = geometry
        missing = [r.cid for r in records if r.cid not in geometry.by_cid]
        if missing:
            raise RuntimeError(f"geometry cache misses {len(missing)} IDs: {missing[:10]}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        data = self.geometry.get(self.geometry.by_cid[record.cid])
        data.y = torch.tensor([record.y], dtype=torch.float32)
        data.text_emb = record.text.reshape(1, 768)
        data.cid = record.cid
        return data
