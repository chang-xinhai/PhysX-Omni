#!/usr/bin/env python3
"""Filter a JSONL benchmark manifest by method, dataset, metric, and readiness."""

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                out[key] = value
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--metrics", nargs="*", default=[])
    parser.add_argument("--ready-only", action="store_true")
    args = parser.parse_args()

    methods = set(args.methods)
    datasets = set(args.datasets)
    metrics = {x.lower() for x in args.metrics}

    rows = []
    for row in read_jsonl(args.input_jsonl):
        if methods and row.get("method") not in methods:
            continue
        if datasets and row.get("dataset") not in datasets:
            continue
        if metrics and str(row.get("metric") or "").lower() not in metrics:
            continue
        if args.ready_only and not bool(row.get("ready")):
            continue
        rows.append(row)

    write_jsonl(rows, args.output_jsonl)
    if args.output_csv:
        write_csv(rows, args.output_csv)

    ready = sum(bool(row.get("ready")) for row in rows)
    statuses = {}
    for row in rows:
        for token in str(row.get("status") or "").split(";"):
            token = token.strip()
            if token:
                statuses[token] = statuses.get(token, 0) + 1
    print(json.dumps({"rows": len(rows), "ready": ready, "statuses": statuses}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
