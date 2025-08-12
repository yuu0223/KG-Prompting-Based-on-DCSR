import json
import os
import csv
import re
import random
from tqdm import tqdm

### --- new file
with open('../output/Cal_Paths_Nodes/MindMap_Path_Only.csv', 'w', newline='') as f4: ###
    writer = csv.writer(f4)
    writer.writerow(['Q_ID','Avg_Distance_All', 'Avg_Distance_Path', 'Avg_Distance_Nei'])


### --- Start!
with open("../data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

with open("../output/MindMap/Final_Q714_Gemini1.5/20250505_Gemini15.json", "r", encoding='utf-8') as f1: ###
    data = json.load(f1)

round_count = 0
# skip_id = [2574, 4405, 938, 4409]

for line in tqdm(lines[round_count:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
    x = json.loads(line)

    id = x["Q_ID"]
    # if id in skip_id:
    #     continue

    result = [item for item in data if item.get("Q_ID") == id]
    print("Q_ID from paths json:", result[0]["Q_ID"])

    ### --- Path-based
    # paths_list = result[0]["path-based"] # list
    paths_list_path = [q.replace(" ", "_") for q in result[0]["path-based"]]
    print("paths_list:", paths_list_path)

    # --- 提取 entities
    exclude_relations = {"has_symptom", "can_check_disease", "possible_disease", "need_medical_test", "possible_cure_disease", "need_medication"}
    entity_paths_path = []
    for path in paths_list_path:
        ### --- for mindmap
        parts = path.split('->')
        entities = [e for e in parts if e not in exclude_relations]
        entity_paths_path.append(entities)

    entity_paths_path_processd = []
    for path in entity_paths_path:
        [entity_paths_path_processd.append(entities) for entities in path]

    # 建立所有 paths 中所有實體的總集合
    all_path_entities_path = set(entity_paths_path_processd)
    print("all_path_entities_path:", len(all_path_entities_path))



    ### --- Neighbor-based
    paths_list_nei = [q.replace(" ", "_") for q in result[0]["neighbor-based"]]
    print("paths_list:", paths_list_nei)

    # --- 提取 entities
    exclude_relations = {"has_symptom", "can_check_disease", "possible_disease", "need_medical_test", "possible_cure_disease", "need_medication"}
    entity_paths_nei = []
    for path in paths_list_nei:
        ### --- for mindmap
        parts = path.split('->')
        entities = [e for e in parts if e not in exclude_relations]
        entity_paths_nei.append(entities)
        
    print(entity_paths_nei)
    # neighbor-based 要另外處理
    entity_paths_nei_processd = []
    for path in entity_paths_nei:
        entity_paths_nei_processd.append(path[0])
        sep = path[1].split(',')
        [entity_paths_nei_processd.append(entities) for entities in sep]

    # 建立所有 paths 中所有實體的總集合
    all_path_entities_nei = set(entity_paths_nei_processd)
    print("all_path_entities_nei:", len(all_path_entities_nei))



    ### --- Path-based + Neighbor-based
    all_paths_list = entity_paths_path_processd + entity_paths_nei_processd
    print("all_paths_list:", all_paths_list)

    ### ---  每條paths不一定都包含所有QN，過濾掉沒出現的QN，避免後續計算出錯
    # 建立所有 paths 中所有實體的總集合
    all_path_entities = set(all_paths_list)
    print("all_path_entities:", len(all_path_entities))
    


    with open('../output/Cal_Paths_Nodes/MindMap_Path_Only.csv', 'a+', newline='') as f6: ###
        writer = csv.writer(f6)
        writer.writerow([result[0]["Q_ID"],len(all_path_entities) ,len(all_path_entities_path), len(all_path_entities_nei)])
        f6.flush()