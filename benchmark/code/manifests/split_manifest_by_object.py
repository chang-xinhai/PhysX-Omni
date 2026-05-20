#!/usr/bin/env python3
"""Split a JSONL manifest into balanced shards while keeping object rows together."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def natural_key(value: str):
    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part for part in parts]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def shard_rows(rows, num_shards: int):
    groups = {}
    for row in rows:
        key = (
            str(row.get("dataset") or ""),
            str(row.get("object_id") or row.get("sample_id") or ""),
        )
        groups.setdefault(key, []).append(row)

    ordered = sorted(groups.items(), key=lambda item: (item[0][0], natural_key(item[0][1])))
    shards = [[] for _ in range(num_shards)]
    for idx, (_key, group_rows) in enumerate(ordered):
        shards[idx % num_shards].extend(group_rows)
    return shards


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--summary-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")

    rows = load_jsonl(args.input_jsonl)
    shards = shard_rows(rows, args.num_shards)
    summary = []
    for idx, shard in enumerate(shards):
        out_path = args.output_prefix.parent / f"{args.output_prefix.name}_{idx:02d}.jsonl"
        write_jsonl(shard, out_path)
        object_keys = {
            (str(row.get("dataset") or ""), str(row.get("object_id") or row.get("sample_id") or ""))
            for row in shard
        }
        ready = sum(1 for row in shard if bool(row.get("ready")))
        missing = len(shard) - ready
        summary.append(
            {
                "shard": idx,
                "jsonl": str(out_path),
                "rows": len(shard),
                "objects": len(object_keys),
                "ready_rows": ready,
                "missing_zero_rows": missing,
            }
        )

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["shard", "jsonl", "rows", "objects", "ready_rows", "missing_zero_rows"],
            )
            writer.writeheader()
            writer.writerows(summary)

    print(json.dumps({"input_rows": len(rows), "num_shards": args.num_shards, "shards": summary}, indent=2))


if __name__ == "__main__":
    main()
