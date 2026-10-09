from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

from pymatgen.core import Structure
from robocrys import StructureCondenser, StructureDescriber

from embed_descriptions import DEFAULT_MODEL_NAME, embed_description_csv


DEFAULT_DATASET_DIR = Path("data/test")
DEFAULT_DESCRIPTION_OUTPUT = Path("outputs/cif_descriptions.csv")
DEFAULT_EMBEDDING_OUTPUT = Path("outputs/matbert_embeddings.pt")


def iter_cif_files(dataset_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.cif" if recursive else "*.cif"
    return sorted(dataset_dir.glob(pattern))


def describe_cif(
    cif_path: Path,
    condenser: StructureCondenser,
    describer: StructureDescriber,
) -> str:
    structure = Structure.from_file(cif_path)
    condensed_structure = condenser.condense_structure(structure)
    return describer.describe(condensed_structure)


def default_errors_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_errors.csv")


def write_descriptions(
    cif_paths: list[Path],
    output_path: Path,
    errors_path: Path,
    strict: bool = False,
    progress_every: int = 25,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    condenser = StructureCondenser()
    describer = StructureDescriber()
    success_count = 0
    error_count = 0

    with output_path.open("w", newline="", encoding="utf-8") as output_file, \
            errors_path.open("w", newline="", encoding="utf-8") as error_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["cif_id", "cif_path", "description"],
        )
        error_writer = csv.DictWriter(
            error_file,
            fieldnames=["cif_id", "cif_path", "error"],
        )
        writer.writeheader()
        error_writer.writeheader()

        for index, cif_path in enumerate(cif_paths, start=1):
            try:
                description = describe_cif(cif_path, condenser, describer)
            except Exception as exc:
                if strict:
                    raise
                error_count += 1
                error_writer.writerow({
                    "cif_id": cif_path.stem,
                    "cif_path": str(cif_path.resolve()),
                    "error": str(exc),
                })
                continue

            success_count += 1
            writer.writerow({
                "cif_id": cif_path.stem,
                "cif_path": str(cif_path.resolve()),
                "description": description,
            })

            if progress_every > 0 and (
                success_count % progress_every == 0
                or index == len(cif_paths)
            ):
                print(
                    f"Generated {success_count}/{len(cif_paths)} descriptions "
                    f"({error_count} errors)"
                )

    return success_count, error_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert CIF files to robocrys descriptions, then "
            "optionally embed them with MatBERT."
        )
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(DEFAULT_DATASET_DIR),
        help="Dataset directory containing CIF files",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_DESCRIPTION_OUTPUT),
        help="CSV path for generated CIF descriptions",
    )
    parser.add_argument(
        "--errors-output",
        default=None,
        help="CSV path for CIF files that fail description generation",
    )
    parser.add_argument(
        "--embedding-output",
        default=str(DEFAULT_EMBEDDING_OUTPUT),
        help="Path to save MatBERT embeddings",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Only generate descriptions; do not run MatBERT",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan CIF files directly under dataset_dir",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N CIF files; useful for testing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately if one CIF file fails",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N successful CIF descriptions",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show pymatgen/robocrys warnings during description generation",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local model path for MatBERT",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific device such as cuda:0",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    errors_path = (
        Path(args.errors_output)
        if args.errors_output
        else default_errors_path(output_path)
    )

    if not args.show_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    cif_paths = iter_cif_files(dataset_dir, recursive=not args.non_recursive)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than 0")
        cif_paths = cif_paths[:args.limit]

    if not cif_paths:
        raise FileNotFoundError(f"No CIF files found in {dataset_dir}")

    print(f"Found {len(cif_paths)} CIF files under {dataset_dir}")
    success_count, error_count = write_descriptions(
        cif_paths,
        output_path,
        errors_path,
        strict=args.strict,
        progress_every=args.progress_every,
    )
    print(
        f"Saved {success_count} descriptions to {output_path} "
        f"({error_count} errors logged to {errors_path})"
    )

    if args.no_embed:
        return

    result = embed_description_csv(
        output_path,
        args.embedding_output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    print(
        "Saved {num_texts} MatBERT embeddings to {output_path} "
        "with shape {embedding_shape} on {device}".format(**result)
    )


if __name__ == "__main__":
    main()
