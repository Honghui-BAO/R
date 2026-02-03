import argparse
import json
import math
from tqdm import tqdm
import numpy as np

def gao(path, item_path):
    print("Path:", path)
    print("Item Path:", item_path)

    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]

    CC = 0

    with open(f"{item_path}.txt", 'r') as f:
        items = f.readlines()

    item_names = [_.split('\t')[0].strip("\"").strip().strip('\n').strip('\"') for _ in items]
    item_ids = list(range(len(item_names)))
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:
            item_dict[item_names[i]].append(item_ids[i])

    ALLNDCG = np.zeros(2)  # for topk 5 and 10
    ALLHR = np.zeros(2)

    topk_list = [5, 10]

    for p in path:
        print(f"Processing: {p}")
        with open(p, 'r') as f:
            test_data = json.load(f)

        text = [[_.strip().strip("\"") for _ in sample["predict"]] for sample in test_data]

        for index, sample in tqdm(enumerate(text), total=len(text)):
            if isinstance(test_data[index]['output'], list):
                target_item = test_data[index]['output'][0].strip().strip("\"")
            else:
                target_item = test_data[index]['output'].strip().strip("\"")

            minID = 1000000

            for i in range(len(sample)):
                if sample[i] not in item_dict:
                    CC += 1
                if sample[i] == target_item:
                    minID = i

            for idx, topk in enumerate(topk_list):
                if minID < topk:
                    ALLNDCG[idx] += (1 / math.log(minID + 2))
                    ALLHR[idx] += 1

    print("NDCG:", ALLNDCG / len(text) / (1.0 / math.log(2)))
    print("HR:", ALLHR / len(text))
    print("Missing items:", CC)


def main():
    parser = argparse.ArgumentParser(description="Run gao evaluation.")
    parser.add_argument('--path', nargs='+', required=True, help='Path(s) to result JSON file(s).')
    parser.add_argument('--item_path', required=True, help='Path to item txt file.')

    args = parser.parse_args()

    gao(path=args.path, item_path=args.item_path)

if __name__ == "__main__":
    main()
