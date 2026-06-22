import json
import gzip
from tqdm import tqdm
import glob


def main():
    print("main")
    files_list = glob.glob("/dccstor/retrieve-rerank2/data/embeddings-training-data/simlm/*.gz")
    name_list = [i.split("/")[-1]for i in files_list]

    for file_name,name in zip(files_list,name_list):
        print(file_name)
        out_file = open("/dccstor/retrieve-rerank2/irl/data/unsup/"+name+".jsonl","w")
        with gzip.open(file_name,"rb") as fp:
            for line in tqdm(fp):
                json_data = json.loads(line)
                json.dump({"query":json_data["query"],"pos":json_data["positives"]["docs"],"neg":[]},out_file)
                out_file.write("\n")
        out_file.close()
    # file_name = "/dccstor/retrieve-rerank2/data/embeddings-training-data/simlm/hard_negatives/msmarco_train_w_docs.jsonl.gz"
    # out_file = open("/dccstor/retrieve-rerank2/irl/data/unsup/msmarco.jsonl","w")
    # with gzip.open(file_name,"rb") as fp:
    #     for line in tqdm(fp):
    #         json_data = json.loads(line)
    #         json.dump({"query":json_data["query"],"pos":json_data["positives"]["docs"],"neg":[]},out_file)
    #         out_file.write("\n")
    #     out_file.close()
            
    
    
if __name__=="__main__":
    main()