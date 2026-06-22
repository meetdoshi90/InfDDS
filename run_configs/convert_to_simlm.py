import json
import gzip
from datasets import load_dataset 

#IN_PATH = "/dccstor/retrieve-rerank2/data/embeddings-training-data/sentence_transformers/"
IN_PATH = "/dccstor/retrieve-rerank2/data/embeddings-training-data/simlm/"
OUT_PATH = "/dccstor/sdg/meet/data/DDS/sent-trans/"

DATA = "WikiAnswers_pairs.jsonl.gz"

processed_data = []

with gzip.open(IN_PATH + DATA, "rt", encoding="utf-8") as fin:
    for line in fin:
        data = json.loads(line)
        print(data)
        query = data[0]
        pos = data[1] 
        processed = {
            "query": query,
            "pos": [pos]
        }
        processed_data.append(processed)
print(len(processed_data))
with gzip.open(OUT_PATH + DATA, "wt", encoding="utf-8") as fout:
    for item in processed_data:
        fout.write(json.dumps(item) + "\n")

dataset = load_dataset('json', data_files=OUT_PATH+DATA, split='train')