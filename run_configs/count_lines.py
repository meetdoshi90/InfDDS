import gzip
import json
from datasets import load_dataset
from tqdm import tqdm

file_path = "/dccstor/sdg/meet/data/TableRetrieval/OTT-QA/train_nomarker/train.jsonl.gz"

#dataset = load_dataset('json', data_files=file_path, split='train')

with gzip.open(file_path, "rt", encoding="utf-8") as f:
    for line in f:
        print(json.loads(line))
        break

with gzip.open(file_path, "rt", encoding="utf-8") as f:
    line_count = sum(1 for _ in tqdm(f))


print(f"Number of lines: {line_count}")