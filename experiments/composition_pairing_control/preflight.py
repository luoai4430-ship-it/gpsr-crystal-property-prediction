"""Strict, label-free preflight for frozen composition-conditioned pairing.

This module uses only the Python standard library. It does not load models,
embeddings, property targets, or checkpoints, and does not mutate source data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
    "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
    "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es "
    "Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
TASK_PATHS = {
    "Jarvis-Bandgap_MBJ": "jarvis/mbj_bandgap",
    "Jarvis-Bandgap_OPT": "jarvis/optb88vdw_bandgap",
    "Jarvis-FormationEnergy": "jarvis/formation_energy_peratom",
    "Jarvis-BulkModulusKv": "jarvis/bulk_modulus_kv",
    "MP-Bandgap": "mp_2018/MP-bandgap",
    "MP-BulkModuli": "mp_2018_small/bulk",
}
REPEATED_TASKS = (
    "Jarvis-Bandgap_MBJ", "Jarvis-BulkModulusKv",
    "Jarvis-FormationEnergy", "MP-Bandgap",
)
RUNS = tuple((42, task) for task in TASK_PATHS) + tuple(
    (seed, task) for seed in (7, 1234) for task in REPEATED_TASKS
)
SAMPLE_FIELDS = (
    "task", "sample_index", "sample_id", "reduced_composition", "element_set",
    "description_sha256", "formula", "description_id", "match_strategy",
    "group_size_reduced_composition", "group_size_element_set",
    "metadata_formula", "formula_source", "formula_source_file", "formula_source_sha256",
)
COVERAGE_FIELDS = (
    "task", "group_type", "num_samples", "num_eligible", "num_groups_eligible",
    "coverage_fraction", "num_groups", "num_singletons", "max_group_size",
    "eligible_group_size_histogram", "same_description_ordered_pair_fraction",
    "num_eligible_samples_without_distinct_description_peer",
    "fraction_eligible_samples_without_distinct_description_peer",
    "distinct_mapping_space_capped", "mapping_space_cap", "sampling_note",
)


class FormulaError(ValueError):
    pass


def parse_formula(formula: str) -> dict[str, Fraction]:
    """Parse element counts and nested ()/[] groups with exact rational counts.

    Counts accept positive integers, decimals, and explicit rational a/b forms.
    Leading multipliers, charges, hydrate separators, isotope notation, and
    unknown elements fail explicitly instead of being silently simplified.
    """
    if not isinstance(formula, str):
        raise FormulaError("formula must be a string")
    text = re.sub(r"\s+", "", formula)
    if not text:
        raise FormulaError("empty formula")
    pos = 0

    def count() -> Fraction:
        nonlocal pos
        match = re.match(r"(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?", text[pos:])
        if match is None:
            return Fraction(1)
        token = match.group(0)
        pos += len(token)
        if "/" in token:
            numerator, denominator = token.split("/")
            if int(denominator) == 0:
                raise FormulaError("zero denominator")
            result = Fraction(Decimal(numerator)) / int(denominator)
        else:
            result = Fraction(Decimal(token))
        if result <= 0:
            raise FormulaError("element/group counts must be positive")
        return result

    def sequence(closing: str | None = None) -> dict[str, Fraction]:
        nonlocal pos
        values: dict[str, Fraction] = defaultdict(Fraction)
        terms = 0
        while pos < len(text):
            char = text[pos]
            if char in ")]":
                if closing is None or char != closing:
                    raise FormulaError(f"unmatched {char!r} at character {pos}")
                if not terms:
                    raise FormulaError("empty group")
                pos += 1
                return dict(values)
            if char in "([":
                pos += 1
                inner = sequence(")" if char == "(" else "]")
                factor = count()
                for symbol, value in inner.items():
                    values[symbol] += value * factor
                terms += 1
                continue
            match = re.match(r"[A-Z][a-z]?", text[pos:])
            if match is None:
                raise FormulaError(f"invalid character {char!r} at character {pos}")
            symbol = match.group(0)
            if symbol not in ELEMENTS:
                raise FormulaError(f"unknown element {symbol!r}")
            pos += len(symbol)
            values[symbol] += count()
            terms += 1
        if closing is not None:
            raise FormulaError(f"missing closing {closing!r}")
        if not terms:
            raise FormulaError("empty formula")
        return dict(values)

    result = sequence()
    if pos != len(text):
        raise FormulaError(f"unconsumed suffix {text[pos:]!r}")
    return result


def composition_keys(formula: str) -> tuple[str, str]:
    counts = parse_formula(formula)
    denominator = 1
    for value in counts.values():
        denominator = math.lcm(denominator, value.denominator)
    integers = {element: int(value * denominator) for element, value in counts.items()}
    divisor = 0
    for value in integers.values():
        divisor = math.gcd(divisor, value)
    symbols = sorted(integers)
    reduced = "|".join(f"{symbol}:{integers[symbol] // divisor}" for symbol in symbols)
    return reduced, "|".join(symbols)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"missing/duplicate CSV headers: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"malformed CSV row with unexpected fields: {path}")
    return fields, rows


def column(fields: list[str], names: tuple[str, ...], path: Path) -> str:
    found = [name for name in names if name in fields]
    if len(found) != 1:
        raise ValueError(f"expected one of {names}, got {found}: {path}")
    return found[0]


def description_file(data_root: Path, task: str) -> Path:
    base = data_root / TASK_PATHS[task]
    candidates = [path for path in (base / "description.csv", base / "cif/description.csv")
                  if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"description.csv absent under {base}")
    if len(candidates) > 1 and len({sha256(path) for path in candidates}) != 1:
        raise ValueError(f"multiple different description files: {candidates}")
    return candidates[0]


def load_descriptions(path: Path, required_ids=None) -> tuple[dict[str, dict[str, str]], int]:
    fields, rows = read_csv(path)
    id_col = column(fields, ("material_id", "id"), path)
    formula_col = column(fields, ("Composition", "composition"), path)
    text_col = column(fields, ("Description", "description", "text"), path)
    aliases: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        identifier = (row.get(id_col) or "").strip()
        source_id = (row.get('source_cif_id') or '').strip()
        if required_ids is not None and identifier not in required_ids and source_id not in required_ids:
            continue
        formula = (row.get(formula_col) or "").strip()
        description = row.get(text_col)
        if not identifier or not formula or not description or not description.strip():
            raise ValueError(f"empty identity/formula/description: {path}:{index}")
        metadata_formula = formula
        formula_source, formula_file, formula_hash = 'description_metadata', '', ''
        try:
            reduced, elements = composition_keys(formula)
        except (FormulaError, ArithmeticError) as exc:
            # Some upstream CSV formulas were converted into date strings by a
            # spreadsheet. Never guess their chemistry: recover only from the
            # original same-ID CIF's explicit formula-sum tag and keep evidence.
            cif_id = (row.get('source_cif_id') or identifier).strip()
            cif = path.parent / 'cif' / (cif_id + '.cif')
            if not cif.is_file():
                raise ValueError(f"unparseable formula {formula!r}: {path}:{index}: {exc}; original CIF unavailable") from exc
            matches = re.findall(r'^_chemical_formula_sum\s+(.+)\s*$', cif.read_text(encoding='utf-8'), re.MULTILINE)
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one CIF formula-sum tag: {cif}') from exc
            formula = matches[0].strip().strip("'\"")
            reduced, elements = composition_keys(formula)
            formula_source, formula_file, formula_hash = 'original_cif_formula_sum', str(cif.resolve()), sha256(cif)
        entry = {"formula": formula, "description_id": identifier,
                 "metadata_formula": metadata_formula, "formula_source": formula_source,
                 "formula_source_file": formula_file, "formula_source_sha256": formula_hash,
                 "reduced_composition": reduced, "element_set": elements,
                 "description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest()}
        identifiers = [(identifier, "identity")]
        source_id = (row.get("source_cif_id") or "").strip()
        if source_id:
            identifiers.append((source_id, "source_cif_id"))
        for alias, strategy in identifiers:
            record = dict(entry, alias_strategy=strategy)
            old = aliases.get(alias)
            if old is not None:
                compared = ("reduced_composition", "description_sha256")
                if any(old[key] != record[key] for key in compared):
                    raise ValueError(f"conflicting description alias {alias!r}: {path}:{index}")
            else:
                aliases[alias] = record
    if not rows:
        raise ValueError(f"empty descriptions: {path}")
    return aliases, len(rows)


def resolve_id(sample_id: str, aliases: dict[str, dict[str, str]]) -> dict[str, str]:
    if sample_id in aliases:
        return dict(aliases[sample_id], match_strategy=aliases[sample_id]["alias_strategy"])
    # The dataset may preserve multiple property records for one structure.
    match = re.fullmatch(r"(.+)__record_\d+", sample_id)
    if match and match.group(1) in aliases:
        record = aliases[match.group(1)]
        return dict(record, match_strategy="record_suffix+" + record["alias_strategy"])
    raise KeyError(f"test sample identity has no exact description alias: {sample_id!r}")


def derangement_count_capped(n: int, cap: int) -> int:
    previous, current = 1, 0
    for index in range(2, n + 1):
        previous, current = current, min(cap, (index - 1) * (previous + current))
    return current if n else 1


def coverage(task: str, rows: list[dict[str, object]], group_type: str) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_type])].append(row)
    eligible_groups = [items for items in groups.values() if len(items) >= 2]
    eligible = sum(map(len, eligible_groups))
    same_pairs = all_pairs = no_distinct_peer = 0
    mapping_cap = 10**12
    mappings = 1
    for items in eligible_groups:
        n = len(items)
        descriptions = Counter(str(row["description_sha256"]) for row in items)
        all_pairs += n * (n - 1)
        same_pairs += sum(value * (value - 1) for value in descriptions.values())
        if len(descriptions) == 1:
            no_distinct_peer += n
        mappings = min(mapping_cap, mappings * derangement_count_capped(n, mapping_cap))
    if not eligible_groups:
        mappings = 0
    for items in groups.values():
        for row in items:
            row["group_size_" + group_type] = len(items)
    return {
        "task": task, "group_type": group_type, "num_samples": len(rows),
        "num_eligible": eligible, "num_groups_eligible": len(eligible_groups),
        "coverage_fraction": eligible / len(rows), "num_groups": len(groups),
        "num_singletons": sum(len(items) == 1 for items in groups.values()),
        "max_group_size": max(map(len, groups.values()), default=0),
        "eligible_group_size_histogram": json.dumps(dict(sorted(Counter(map(len, eligible_groups)).items()))),
        "same_description_ordered_pair_fraction": same_pairs / all_pairs if all_pairs else None,
        "num_eligible_samples_without_distinct_description_peer": no_distinct_peer,
        "fraction_eligible_samples_without_distinct_description_peer": no_distinct_peer / eligible if eligible else None,
        "distinct_mapping_space_capped": mappings, "mapping_space_cap": mapping_cap,
        "sampling_note": "20 fixed draws; mappings may repeat; draws are not independent new trained models",
    }


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def self_test() -> None:
    expected = {
        "Li3(PO4)2": "Li:3|O:8|P:2",
        "Ca3[Fe(CN)6]2": "C:12|Ca:3|Fe:2|N:12",
        "Fe0.5Mg0.5O": "Fe:1|Mg:1|O:2",
        "Fe1/2Mg1/2O": "Fe:1|Mg:1|O:2",
        "Al4O6": "Al:2|O:3", "Ba1 O3 Ti1": "Ba:1|O:3|Ti:1",
        "CH3COOH": "C:1|H:2|O:1", "Uuo": None,
    }
    for formula, key in expected.items():
        if key is None:
            try:
                composition_keys(formula)
            except FormulaError:
                continue
            raise AssertionError(f"unknown element accepted: {formula}")
        assert composition_keys(formula)[0] == key, formula
    for formula in ("", "Xx2O", "Fe0O", "Ca(OH", "Ca[OH)", "()", "2H2O", "Fe+2", "H2O·5H2O", "Li1/0O", "Li..5O"):
        try:
            composition_keys(formula)
        except (FormulaError, ArithmeticError):
            pass
        else:
            raise AssertionError(f"malformed formula accepted: {formula!r}")
    assert [derangement_count_capped(n, 1000) for n in range(2, 7)] == [1, 2, 9, 44, 265]
    print("SELF_TEST_PASS: nested groups, exact decimal/rational reduction, element whitelist, invalid formulas, mapping counts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pairing-root", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.data_root is None and args.pairing_root is None:
            return 0
    if args.data_root is None or args.pairing_root is None:
        parser.error("--data-root and --pairing-root are required unless only running --self-test")
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "RUNNING", "protocol": "composition-conditioned frozen pairing v1",
        "num_required_runs": len(RUNS), "num_permutations_each": 20,
        "primary_group_type": "reduced_composition", "secondary_group_type": "element_set",
        "labels_used_for_grouping": False, "source_files": [],
        "sample_groups_schema": list(SAMPLE_FIELDS), "group_coverage_schema": list(COVERAGE_FIELDS),
    }

    def source(path: Path) -> None:
        report["source_files"].append({"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size})

    try:
        ids_by_task: dict[str, list[str]] = {}
        run_records = []
        for seed, task in RUNS:
            path = args.pairing_root / "runs" / f"seed_{seed}" / task / "sample_metrics.csv"
            source(path)
            fields, records = read_csv(path)
            if "sample_id" not in fields or not records:
                raise ValueError(f"missing sample_id or empty test rows: {path}")
            ids = [(row.get("sample_id") or "").strip() for row in records]
            if any(not identifier for identifier in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"empty/duplicate sample IDs: {path}")
            if "task" in fields and any(row["task"] != task for row in records):
                raise ValueError(f"unexpected task in {path}")
            if "seed" in fields and any(row["seed"] != str(seed) for row in records):
                raise ValueError(f"unexpected seed in {path}")
            if task in ids_by_task and ids != ids_by_task[task]:
                raise ValueError(f"test IDs/order differ across training seeds for {task}")
            ids_by_task[task] = ids
            run_records.append({"task": task, "seed": seed, "num_samples": len(ids), "id_order_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()})
        aliases_by_task, description_counts = {}, {}
        for task in TASK_PATHS:
            path = description_file(args.data_root, task)
            source(path)
            required = {re.sub(r'__record_\d+$', '', cid) for cid in ids_by_task[task]}
            aliases_by_task[task], description_counts[task] = load_descriptions(path, required)
        recovered = {}
        for task, aliases in aliases_by_task.items():
            for record in aliases.values():
                if record['formula_source_file']:
                    recovered[record['formula_source_file']] = dict(task=task,
                        description_id=record['description_id'], metadata_formula=record['metadata_formula'],
                        recovered_formula=record['formula'], sha256=record['formula_source_sha256'])
        for path in sorted(recovered):
            source(Path(path))
        report['formula_metadata_recoveries'] = recovered
        sample_rows: list[dict[str, object]] = []
        coverage_rows = []
        for task in TASK_PATHS:
            task_rows = []
            for index, identifier in enumerate(ids_by_task[task]):
                entry = resolve_id(identifier, aliases_by_task[task])
                task_rows.append(dict(entry, task=task, sample_index=index, sample_id=identifier))
            for group_type in ("reduced_composition", "element_set"):
                coverage_rows.append(coverage(task, task_rows, group_type))
            sample_rows.extend(task_rows)
        report.update(status="PASS", num_runs=len(run_records), num_tasks=len(TASK_PATHS),
                      num_unique_task_samples=len(sample_rows), runs=run_records,
                      description_row_counts=description_counts,
                      no_missing_test_id=True, identical_test_id_order_across_seeds=True,
                      composition_parser="strict element whitelist; exact Decimal/Fraction counts; gcd reduced stoichiometry",
                      warnings=["Coverage is subset-specific; repeated mappings do not supply independent evidence.",
                                "Element-set matching preserves element identity but may change stoichiometry.",
                                "Grouping reads identity, composition, and description only; no target values are used."])
        write_csv(args.out / "sample_groups.csv", SAMPLE_FIELDS, sample_rows)
        write_csv(args.out / "group_coverage.csv", COVERAGE_FIELDS, coverage_rows)
        report["output_hashes"] = {name: sha256(args.out / name) for name in ("sample_groups.csv", "group_coverage.csv")}
    except Exception as exc:
        report.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        (args.out / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"PREFLIGHT_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    (args.out / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs": len(RUNS), "tasks": len(TASK_PATHS), "coverage": coverage_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
