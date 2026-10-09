#!/usr/bin/env python3
"""Recreate the paper split and record it by persistent sample ID."""

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("id_prop", type=Path, help="Headerless sample_id,target CSV")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.valid_ratio <= 0:
        raise ValueError("Ratios must be positive")
    if args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than one")

    rows = []
    with args.id_prop.open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.reader(handle)):
            if len(row) < 2:
                raise ValueError(f"Row {row_index + 1} has fewer than two columns")
            rows.append({"sample_id": row[0].strip(), "target": row[1].strip(), "source_index": row_index})

    random.seed(args.seed)
    random.shuffle(rows)
    n_train = int(len(rows) * args.train_ratio)
    n_valid = int(len(rows) * args.valid_ratio)
    for index, row in enumerate(rows):
        row["shuffled_index"] = index
        if index < n_train:
            row["split"] = "train"
        elif index < n_train + n_valid:
            row["split"] = "valid"
        else:
            row["split"] = "test"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "target", "source_index", "shuffled_index", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows: train={n_train}, valid={n_valid}, test={len(rows)-n_train-n_valid}")


if __name__ == "__main__":
    main()
