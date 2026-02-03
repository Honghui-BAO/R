# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# import transformers
# import torch
import os
import fire
import math
import json
import pandas as pd
import numpy as np
    
from tqdm import tqdm

# /storage_fast/hhbao/project/R3/output_dir/CDs_and_Vinyl/latent/checkpoint-770/final_result.json
# def gao(path = ['/storage_fast/hhbao/project/R3/output_dir/CDs_and_Vinyl/latent/checkpoint-770/final_result.json'\
#                 ,'/storage_fast/hhbao/project/R3/output_dir/CDs_and_Vinyl/nomul_attn_lr_1e-4/final_result.json'], item_path='/storage_fast/hhbao/project/R3/data/info/CDs_and_Vinyl_5_2015-10-2018-11.txt'):
def gao(path, item_path):

    print(path)
    print(item_path)
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC=0

    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    item_names = [_.split('\t')[0].strip("\"").strip(" ").strip('\n').strip('\"') for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    # print(item_dict) 
    ALLNDCG = np.zeros(2) # 1 3 5 10 20
    ALLHR = np.zeros(2)

    result_dict = dict()
    topk_list = [5,10]
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
        }
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        
        text = [ [_.strip(" \n").strip("\"").strip(" ") for _ in sample["predict"]] for sample in test_data]
        
        for index, sample in tqdm(enumerate(text)):

                if type(test_data[index]['output']) == list:
                    target_item = test_data[index]['output'][0].strip("\"").strip(" ")
                else:
                    target_item = test_data[index]['output'].strip(" \n\"")
                minID = 1000000
                # rank = dist.argsort(dim = -1)
                for i in range(len(sample)):
                    # for _ in item_dict[target_item]:
                        # if rank[i][0] == _:
                            # minID = i
                    
                    if sample[i] not in item_dict:
                        CC += 1
                        # print(sample[i], index)
                    if sample[i] == target_item:
                        minID = i
                for index, topk in enumerate(topk_list):
                    if minID < topk:
                        ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                        ALLHR[index] = ALLHR[index] + 1
        print("---HR---")
        print(ALLHR / len(text))
        print("---NDCG---")
        print(ALLNDCG / len(text) / (1.0 / math.log(2)))
        print(CC)

if __name__=='__main__':
    fire.Fire(gao)
