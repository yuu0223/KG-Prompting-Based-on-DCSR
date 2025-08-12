import json
import os
import csv
import re
import random
from tqdm import tqdm

### --- new file
with open('../output/Cal_Paths_Distance/Non-QN/StienerTree_PR.csv', 'w', newline='') as f4: ###
    writer = csv.writer(f4)
    writer.writerow(['Q_ID','Distance'])


### --- Start!
with open("../data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

with open("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/SteinerTree/Final_Q714_Gemini1.5/20250505_PR_Gemini15.json", "r", encoding='utf-8') as f1: ###
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
    query_nodes = result[0]["query_nodes"]

    ### --- 提取 path entities
    entity_paths = []
    for path in paths_list:
        parts = path.split('->')
        entities = parts[::2]  # 每隔兩個取一次（0, 2, 4, ...）
        entity_paths.append(entities)
        
    print("entity_paths", entity_paths)

    ### --- 開始計算 Non-QN 距離
    # 1. 找出所有非 QNs 的節點
    non_qns_nodes = set()
    for path in entity_paths:
        for node in path:
            if node not in query_nodes:
                non_qns_nodes.add(node)

    non_qns_nodes = list(non_qns_nodes)

    # 2. 計算這些非 QNs 節點兩兩之間的距離（只看在同一條路徑內的）
    from itertools import combinations

    total_distance = 0
    count = 0

    for path in entity_paths:
        node_positions = {node: idx for idx, node in enumerate(path) if node in non_qns_nodes}
        
        for n1, n2 in combinations(node_positions.keys(), 2):
            dist = abs(node_positions[n1] - node_positions[n2])
            total_distance += dist
            count += 1

    avg_distance = total_distance / count if count else 0
    print(f"平均距離: {round(avg_distance,3)}")
    

    with open('../output/Cal_Paths_Distance/Non-QN/StienerTree_PR.csv', 'a+', newline='') as f6: ###
        writer = csv.writer(f6)
        writer.writerow([result[0]["Q_ID"], round(avg_distance,3)])
        f6.flush()


