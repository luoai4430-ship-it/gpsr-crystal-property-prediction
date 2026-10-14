#!/usr/bin/env python3
"""Verify release checksums, required files, and canonical result summaries."""

from __future__ import annotations

import hashlib
import importlib.util
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums() -> int:
    checked = 0
    for line in (ROOT / "metadata/checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"missing checksummed file: {relative}")
        if sha256(path) != expected:
            raise AssertionError(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def verify_required_files() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "model/semantic_residual.py",
        "experiments/final_10task/summary/multiseed_paired_results.csv",
        "experiments/backbone_transfer_alignn/summary/alignn_multiseed_paired.csv",
        "experiments/cartnet_gpsr/summary/cartnet_seed1_10task.csv",
        "experiments/final_confirmatory/semantic_intervention/macro_summary.csv",
        "experiments/semantic_pairing_permutation/task_summary.csv",
        "experiments/semantic_pairing_permutation/validation.json",
        "experiments/semantic_pairing_permutation/output_correction_analysis/report.md",
        "experiments/semantic_pairing_permutation/output_correction_analysis/run_summary.csv",
        "experiments/semantic_pairing_permutation/output_correction_analysis/aggregate.json",
        "experiments/composition_pairing_control/preflight.json",
        "experiments/composition_pairing_control/group_coverage.csv",
        "experiments/composition_pairing_control/task_summary.csv",
        "experiments/composition_pairing_control/aggregate.json",
        "experiments/composition_pairing_control/validation.json",
        "experiments/composition_pairing_control/report.md",
        "docs/EXPERIMENT_PROCESS_ARCHIVE.md",
        "metadata/server_experiment_archive_provenance.json",
        "metadata/server_experiment_process_manifest.csv",
        "metadata/server_experiment_excluded_artifacts.csv",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise AssertionError(f"missing release files: {missing}")


def verify_process_archive() -> int:
    """Verify every retained server experiment file against its source hash."""
    manifest = ROOT / "metadata/server_experiment_process_manifest.csv"
    checked = 0
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = Path(row["path"])
            if not source.parts or source.parts[0] != "experiments":
                raise AssertionError(f"unexpected process-manifest path: {source}")
            relative = Path("experiments/process_archive", *source.parts[1:])
            path = ROOT / relative
            if not path.is_file():
                raise AssertionError(f"missing archived process file: {relative}")
            if path.stat().st_size != int(row["bytes"]):
                raise AssertionError(f"size mismatch in process archive: {relative}")
            if sha256(path) != row["sha256"]:
                raise AssertionError(f"checksum mismatch in process archive: {relative}")
            checked += 1
    if checked != 1132:
        raise AssertionError(f"expected 1132 archived process files, found {checked}")
    return checked


def verify_results() -> None:
    path = ROOT / "scripts/audit_release_results.py"
    spec = importlib.util.spec_from_file_location("audit_release_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.audit_sft(False)
    module.audit_alignn(False)
    module.audit_semantic_pairing()
    module.audit_output_correction()
    module.audit_composition_pairing()


def main() -> None:
    verify_required_files()
    verify_results()
    process_files = verify_process_archive()
    checked = verify_checksums()
    print(
        "Release verification passed: canonical results valid, "
        f"{process_files} process files and {checked} release checksums verified"
    )


if __name__ == "__main__":
    main()
