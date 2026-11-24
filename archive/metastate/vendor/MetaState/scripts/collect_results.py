#!/usr/bin/env python3
"""Parse a single lm-eval-harness .out file and append results to a CSV."""
import argparse
import csv
import os
import sys

# -- Key columns used for deduplication --
KEY_COLS = [
    "Model", "Dataset", "Filter", "Metric",
    "n-shot", "steps", "max_new_tokens", "limit",
    "use_cache", "threshold", "block_size",
]

ALL_COLS = [
    "Model", "Dataset", "Filter", "Metric", "Value", "Stderr",
    "n-shot", "steps", "max_new_tokens", "limit",
    "use_cache", "threshold", "block_size",
]


def parse_result_table(filepath):
    """Parse the lm-eval-harness pipe-delimited result table from an .out file.

    Table format (fixed columns separated by |):
      | Tasks | Version | Filter | n-shot | Metric | ↑/↓ | Value | ± | Stderr |
    Index after split:
      [0]=""  [1]=task  [2]=ver  [3]=filter [4]=n-shot [5]=metric [6]=arrow [7]=value [8]=± [9]=stderr
    Continuation rows have empty [1] (task) and [2] (version).
    """
    rows = []
    in_table = False
    last_task = ""
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if not s.startswith("|"):
                in_table = False
                continue
            if "Version" in s and "Metric" in s:
                in_table = True
                continue
            if "---" in s:
                continue
            if in_table:
                parts = [p.strip() for p in s.split("|")]
                if len(parts) < 10:
                    continue
                task = parts[1] or last_task
                if parts[1]:
                    last_task = parts[1]
                rows.append({
                    "Dataset": task,
                    "Filter": parts[3],
                    "Metric": parts[5],
                    "Value": parts[7],
                    "Stderr": parts[9],
                })
    return rows


def append_to_csv(csv_path, new_rows):
    """Append rows to CSV, deduplicating by KEY_COLS."""
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            existing = list(csv.DictReader(f))

    def make_key(row):
        return tuple(row.get(c, "") for c in KEY_COLS)

    index = {make_key(r): i for i, r in enumerate(existing)}

    for row in new_rows:
        k = make_key(row)
        if k in index:
            existing[index[k]] = row
        else:
            existing.append(row)
            index[k] = len(existing) - 1

    existing.sort(
        key=lambda r: (
            r.get("Model", ""),
            r.get("Dataset", ""),
            r.get("Metric", ""),
        )
    )

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLS)
        w.writeheader()
        w.writerows(existing)


def main():
    parser = argparse.ArgumentParser(
        description="Parse lm-eval-harness .out file and append to CSV"
    )
    parser.add_argument("out_file", help="Path to .out file")
    parser.add_argument("csv_file", help="Path to output CSV")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fewshot", default="")
    parser.add_argument("--steps", default="")
    parser.add_argument("--max_new_tokens", default="")
    parser.add_argument("--limit", default="")
    parser.add_argument("--use_cache", default="")
    parser.add_argument("--threshold", default="")
    parser.add_argument("--block_size", default="")
    args = parser.parse_args()

    if not os.path.exists(args.out_file):
        print(f"Error: {args.out_file} not found.", file=sys.stderr)
        sys.exit(1)

    table_rows = parse_result_table(args.out_file)
    if not table_rows:
        print(f"Error: No result table found in {args.out_file}", file=sys.stderr)
        sys.exit(1)

    full_rows = []
    for row in table_rows:
        row.update({
            "Model": args.model,
            "n-shot": args.fewshot,
            "steps": args.steps,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "use_cache": args.use_cache,
            "threshold": args.threshold,
            "block_size": args.block_size,
        })
        # Use task name from table if present, fall back to CLI arg
        if not row.get("Dataset"):
            row["Dataset"] = args.task
        full_rows.append(row)

    append_to_csv(args.csv_file, full_rows)
    print(f"Appended {len(full_rows)} results to {args.csv_file}")


if __name__ == "__main__":
    main()
