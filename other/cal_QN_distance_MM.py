import json
import os
import csv
import re
import random
from tqdm import tqdm

### --- new file
with open('../output/Cal_Paths_Distance/E1/User_GreedyDist_BFS.csv', 'w', newline='') as f4: ###
    writer = csv.writer(f4)
    writer.writerow(['Q_ID','Avg_Distance'])


### --- Start!
with open("../data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

with open("../output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/20250505_BFS_Gemini15.json", "r", encoding='utf-8') as f1: ###
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

    # paths_list = result[0]["path-based"] # list
    paths_list = [q.replace(" ", "_") for q in result[0]["path-based"]]
    print("paths_list:", paths_list)


    # --- 提取 entities
    exclude_relations = {"has_symptom", "can_check_disease", "possible_disease", "need_medical_test", "possible_cure_disease", "need_medication"}
    entity_paths = []
    for path in paths_list:
        ### --- for mindmap
        parts = path.split('->')
        entities = [e for e in parts if e not in exclude_relations]
        entity_paths.append(entities)
        
    print(entity_paths)

    # 過濾掉不含所有 query_nodes 的 path
    query_nodes = result[0]["query_nodes"]
    print("query_nodes:", query_nodes)
    # 建立所有 paths 中所有實體的總集合
    all_path_entities = set(e for path in entity_paths for e in path)
    # 過濾 query_nodes，只保留出現在 paths 裡的
    query_nodes = [qn for qn in query_nodes if qn in all_path_entities]
    print("query_nodes:", query_nodes)


    # --- 計算每條 paths 的平均距離
    if len(query_nodes) == 1:
        avg_dis = 1
        print("only one entity:", entity_paths)
    elif len(paths_list) == 0:
        avg_dis = 0
        print("no paths:", entity_paths)
    else:
        paths_avg_dis = []
        for path in entity_paths:
            dis = 0
            if len(query_nodes) <= 2:
                for i in range(len(query_nodes)-1):
                    start_idx = path.index(query_nodes[i])
                    end_idx = path.index(query_nodes[i+1])
                    distance = abs(end_idx - start_idx)
                    dis = dis + distance
            else:
                round = 1
                for i in range(len(query_nodes)-1):
                    if round == 1:
                        start_idx = path.index(query_nodes[i])
                        end_idx = path.index(query_nodes[i+1])
                        distance = abs(end_idx - start_idx)
                        dis = dis + distance
                        round+=1
                    else:
                        start_idx = path.index(query_nodes[i])+1
                        end_idx = path.index(query_nodes[i+1])
                        distance = abs(end_idx - start_idx)
                        dis = dis + distance
                        round+=1

            # paths_avg_dis.append(round(dis / (len(query_nodes)-1),3))
            paths_avg_dis.append(dis / (len(query_nodes)-1))

        print("paths_avg_dis:", paths_avg_dis)
        avg_dis = sum(paths_avg_dis) / len(paths_avg_dis)
        
    print("paths_avg_dis:",avg_dis)


    with open('../output/Cal_Paths_Distance/E1/User_GreedyDist_BFS.csv', 'a+', newline='') as f6: ###
        writer = csv.writer(f6)
        writer.writerow([result[0]["Q_ID"], avg_dis])
        f6.flush()