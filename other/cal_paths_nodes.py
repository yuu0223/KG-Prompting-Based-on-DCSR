import json
import os
import csv
import re
import random
from tqdm import tqdm

### --- new file
with open('../output/Cal_Paths_Nodes/E5/100_GreedyDist_PR.csv', 'w', newline='') as f4: ###
    writer = csv.writer(f4)
    writer.writerow(['Q_ID','Nodes_Amount'])


### --- Start!
with open("../data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

with open("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/E5_GreedyDist_100/20250530_PR.json", "r", encoding='utf-8') as f1: ###
    data = json.load(f1)

round_count = 0
# skip_id = [2574, 4405, 938, 4409]

for line in tqdm(lines[round_count:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
    x = json.loads(line)

    id = x["Q_ID"]
    result = [item for item in data if item.get("Q_ID") == id]
    print("Q_ID from paths json:", result[0]["Q_ID"])
    # print(result)

    paths_list = result[0]["paths-list"] # list
    print("paths_list:", paths_list)


    # --- 提取 entities
    entity_paths = []
    for path in paths_list:
        parts = path.split('->')
        entities = parts[::2]  # 每隔兩個取一次（0, 2, 4, ...）
        entity_paths.append(entities)
        
    print(entity_paths)

    ### ---  每條paths不一定都包含所有QN，過濾掉沒出現的QN，避免後續計算出錯
    # 建立所有 paths 中所有實體的總集合
    all_path_entities = set(e for path in entity_paths for e in path)
    print("all_path_entities:", len(all_path_entities))
    


    with open('../output/Cal_Paths_Nodes/E5/100_GreedyDist_PR.csv', 'a+', newline='') as f6: ###
        writer = csv.writer(f6)
        writer.writerow([result[0]["Q_ID"], len(all_path_entities)])
        f6.flush()