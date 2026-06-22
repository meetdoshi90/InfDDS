#!/usr/bin/env python3
"""
crisp_aggregate.py

Read an input config.json (list of {"name": "...", "lines": N, "weight": M})
and aggregate all depth-2 entries into depth-1 entries, grouping by:
    depth1 directory (the first directory under ./) AND filename (e.g. farsi.jsonl).

Output format: same as input, e.g.
[
  { "name": "./15/farsi.jsonl", "lines": 8, "weight": 30 },
  { "name": "./7/arabic.jsonl", "lines": 12, "weight": 5 }
]
"""
import json
import os
from collections import defaultdict

def aggregate(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # key = (depth1_str, filename) -> sums
    agg = defaultdict(lambda: {"lines": 0, "weight": 0})

    for entry in data:
        name = entry.get("name", "")
        lines = int(entry.get("lines", 0))
        weight = int(entry.get("weight", 0))

        if not name:
            continue

        # normalize then split
        norm = os.path.normpath(name)
        parts = norm.split(os.sep)

        # handle optional leading '.' (from "./...")
        start_idx = 1 if parts and parts[0] == "." else 0

        # require at least depth1 + filename (e.g. "./15/farsi.jsonl" -> len(parts)-start_idx >= 2)
        if len(parts) - start_idx >= 2:
            depth1 = parts[start_idx]             # e.g. "15"
            filename = parts[-1]                  # e.g. "farsi.jsonl"
            key = (depth1, filename)
            agg[key]["lines"] += lines
            agg[key]["weight"] += weight
        else:
            # fallback: if path can't be parsed, treat entire name as filename under "." parent
            filename = parts[-1] if parts else name
            key = (".", filename)
            agg[key]["lines"] += lines
            agg[key]["weight"] += weight

    # Build output list in stable sorted order: numeric depth1 if possible, else lexical
    def sort_key(item):
        (depth1, filename), _ = item
        try:
            return (0, int(depth1), filename)  # numeric-first
        except Exception:
            return (1, depth1, filename)       # non-numeric later

    out = []
    for (depth1, filename), sums in sorted(agg.items(), key=sort_key):
        if depth1 == ".":
            out_name = f"./{filename}"            # fallback
        else:
            out_name = f"./{depth1}/{filename}"
        out.append({
            "name": out_name,
            "lines": sums["lines"],
            "weight": sums["weight"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Aggregate depth-2 config entries into depth-1 by language.")
    p.add_argument("input", help="input config.json")
    p.add_argument("output", help="output aggregated config.json")
    args = p.parse_args()
    aggregate(args.input, args.output)
    print(f"Wrote aggregated config to {args.output}")
