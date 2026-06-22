import json
from pprint import pprint
# Path to the uploaded file
file_path = "/dccstor/embedding/meet/DDS/FlagEmbedding/meet_dds_compare/sentence_transformers_l6v2_full.json"

# Read and parse the JSON file again
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Prefix to filter names
prefix = "stackexchange_TitleBody_Answer"

# Sum lines where the name starts with the given prefix
count = sum(item.get("lines", 0) for item in data if prefix in item.get("name", "") and 'math.stackexchange' not in item.get("name", ""))
print('Lines sum', count)
count = sum(item.get("weight", 0) for item in data if prefix in item.get("name", "") and 'math.stackexchange' not in item.get("name", ""))
print('Weight sum', count)
import os

first = os.listdir("/dccstor/retrieve-rerank2/data/embeddings-training-data/sentence_transformers/stackexchange_title_body/")
second = os.listdir(f"/dccstor/retrieve-rerank2/data/embeddings-training-data/sentence_transformers/{prefix}")

f_s = set(first)-set(second)

s_f = set(second)-set(first)

print(f_s)
print()
print(s_f)

f_s = [prefix+'/'+x for x in f_s]
s_f = [prefix+'/'+x for x in s_f]


count = sum(item.get("lines", 0) for item in data if prefix in item.get("name", "") and 'math.stackexchange' not in item.get("name", "") and item.get("name", "") not in f_s)
print('Lines sum', count)
count = sum(item.get("weight", 0) for item in data if prefix in item.get("name", "") and 'math.stackexchange' not in item.get("name", "") and item.get("name", "") not in f_s)
print('Weight sum', count)

s_f = [prefix+'/'+x for x in second]

count = {item.get("name"):item.get("lines", 0) for item in data if item.get("name", "") in s_f}
print('Lines sum')
pprint(count)
print()
count = {item.get("name"):item.get("weight", 0) for item in data if item.get("name", "") in s_f}
print('Weight sum')
pprint(count)

count = sum(item.get("lines", 0) for item in data if 'reddit' not in item.get("name", ""))
print('Lines sum')
pprint(count)
print()
count = sum(item.get("weight", 0) for item in data if 'reddit' not in  item.get("name", ""))
print('Weight sum')
pprint(count)



