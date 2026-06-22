#!/usr/bin/env python3
import os, gzip, json, glob
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm 

def load_config(root):
    """
    Load config.json and return:
      - dict mapping cluster path/lang file relative name -> weight
      - sum of weights
    """
    cfg_path = os.path.join(root, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    weight_map = {}
    total_weight = 0.0
    for entry in cfg:
        name = entry["name"]  # e.g. "./0/langxx.jsonl.gz" or "./5/2/langyy.jsonl.gz"
        w = entry.get("weight", 0)
        weight_map[name] = w
        total_weight += w
    return weight_map, total_weight

def merge_lang_files(root, merged_root):
    """
    For a given root (e.g. "32-clusters"), merge per-language files into merged_root/lang-merged.jsonl.gz;
    add field "weight" = normalized weight of the cluster that file came from.
    """
    weight_map, total_weight = load_config(root)
    os.makedirs(merged_root, exist_ok=True)

    # You need to iterate through all cluster-lang files in root
    # i.e. find all *.jsonl.gz under root subfolders
    pattern = os.path.join(root, "*", "*.jsonl")
    # Also under nested folders if depth>1 (for 1024) you might have two levels: root/*/*/*.jsonl.gz
    pattern2 = os.path.join(root, "*", "*", "*.jsonl")

    files = glob.glob(pattern) + glob.glob(pattern2)
    print(f"Found {len(files)} files in {root} to merge.")

    for fpath in tqdm(files):
        # Determine relative name used in config.json
        # config.json uses entries like "./<path>/<lang>.jsonl.gz"
        # So derive same relative name
        rel = os.path.relpath(fpath, root)
        # Ensure leading "./"
        rel = "./" + rel.replace(os.sep, "/")

        if rel not in weight_map:
            # maybe config uses slightly different path conventions; try alternative matches
            # For example config may have "./<a>/<b>/lang.jsonl.gz"
            # We can try to normalize both
            print(f"WARNING: {rel} not in config weight_map")
            continue

        w = weight_map[rel]
        num_languages = 19
        if total_weight > 0:
            norm_w = w / (total_weight/num_languages)
        else:
            norm_w = 0.0

        # read all lines from this cluster language file, write to merged file
        lang = Path(fpath).stem  # e.g. langxx
        # merged file name
        merged_fname = f"{lang}-merged.jsonl.gz"
        merged_fpath = os.path.join(merged_root, merged_fname)

        # open merged file in append mode so that cluster parts add up
        with open(fpath, "rt") as fin, open(merged_fpath, "at") as fout:
            for line in fin:
                obj = json.loads(line)
                # add weight field
                obj["weight"] = norm_w
                fout.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    roots = ["32-clusters", "1024-clusters"]
    for root in roots:
        merged_root = root + "-merged"
        print(f"Merging files under {root} into {merged_root}")
        merge_lang_files(root, merged_root)
    print("Done merging.")
