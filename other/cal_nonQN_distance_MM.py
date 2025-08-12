import json
import os
import csv
import re
import random
from tqdm import tqdm

### --- new file
with open('../output/Cal_Paths_Distance/Non-QN/MindMap_Path.csv', 'w', newline='') as f4: ###
    writer = csv.writer(f4)
    writer.writerow(['Q_ID','Distance'])


### --- Start!
with open("../data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

with open("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/MindMap/Final_Q714_Gemini1.5/20250505_Gemini15.json", "r", encoding='utf-8') as f1: ###
    data = json.load(f1)

round_count = 0
# skip_id = [2574, 4405, 938, 4409]

for line in tqdm(lines[round_count:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
    x = json.loads(line)

    id = x["Q_ID"]
    result = [item for item in data if item.get("Q_ID") == id]
    print("Q_ID from paths json:", result[0]["Q_ID"])
    # print(result)

    paths_list_path = [q.replace(" ", "_") for q in result[0]["path-based"]]
    print("paths_list:", paths_list_path)
    query_nodes = result[0]["query_nodes"]

    ### --- 提取 path entities
    exclude_relations = {"has_symptom", "can_check_disease", "possible_disease", "need_medical_test", "possible_cure_disease", "need_medication"}
    entity_paths = []
    for path in paths_list_path:
        ### --- for mindmap
        parts = path.split('->')
        entities = [e for e in parts if e not in exclude_relations]
        entity_paths.append(entities)
    
    print("entity_paths:", entity_paths)

    ### ---  刪除MindMap中重複的實體
    cleaned_entity_paths = []
    for path in entity_paths:
        clean_path = [path[0]]
        for i in range(1, len(path)):
            if path[i] != path[i - 1]:
                clean_path.append(path[i])
        cleaned_entity_paths.append(clean_path)

    print("cleaned_entity_paths:", cleaned_entity_paths)

    ### --- 開始計算 Non-QN 距離
    # 1. 找出所有非 QNs 的節點
    non_qns_nodes = set()
    for path in cleaned_entity_paths:
        for node in path:
            if node not in query_nodes:
                non_qns_nodes.add(node)

    non_qns_nodes = list(non_qns_nodes)

    # 2. 計算這些非 QNs 節點兩兩之間的距離（只看在同一條路徑內的）
    from itertools import combinations

    total_distance = 0
    count = 0

    for path in cleaned_entity_paths:
        node_positions = {node: idx for idx, node in enumerate(path) if node in non_qns_nodes}
        
        for n1, n2 in combinations(node_positions.keys(), 2):
            dist = abs(node_positions[n1] - node_positions[n2])
            print(f"Distance between {n1} and {n2}: {dist}")
            total_distance += dist
            count += 1

    avg_distance = total_distance / count if count else 0
    print(f"平均距離: {round(avg_distance,3)}")
    

    with open('../output/Cal_Paths_Distance/Non-QN/MindMap_Path.csv', 'a+', newline='') as f6: ###
        writer = csv.writer(f6)
        writer.writerow([result[0]["Q_ID"], round(avg_distance,3)])
        f6.flush()


