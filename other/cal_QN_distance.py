import json
import os
import csv
import re
import random
from tqdm import tqdm

### --- new file
with open('../output/Cal_Paths_Distance/E2/Path5_GreedyDist_BFS.csv', 'w', newline='') as f4: ###
    writer = csv.writer(f4)
    writer.writerow(['Q_ID','Avg_Distance'])


### --- Start!
with open("../data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

with open("../output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/E2_Path5/20250526_BFS.json", "r", encoding='utf-8') as f1: ###
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
    only_entities = []
    for path in paths_list:
        parts = path.split('->')
        entities = parts[::2]  # 每隔兩個取一次（0, 2, 4, ...）
        only_entities.append(entities)
        
    print(only_entities)

    ### ---  每條paths不一定都包含所有QN，過濾掉沒出現的QN，避免後續計算出錯
    query_nodes = result[0]["query_nodes"]
    print("query_nodes:", query_nodes)
    # 建立所有 paths 中所有實體的總集合
    only_entities_set = set(e for path in only_entities for e in path)
    # 過濾 query_nodes，只保留出現在 paths 裡的
    query_nodes = [qn for qn in query_nodes if qn in only_entities_set]
    print("query_nodes:", query_nodes)


    ### --- 計算每條 paths 的平均距離
    # 只有一個QN時，會找與QN距離1-hop的所有節點，所以距離為1
    if len(query_nodes) == 1:
        avg_dis = 1
        print("only one entity:", only_entities)
    elif len(paths_list) == 0:
        avg_dis = 0
        print("no paths:", only_entities)
    else:
        paths_avg_dis = []
        for path in only_entities:
            dis = 0
            if len(query_nodes) <= 2:
                for i in range(len(query_nodes)-1):
                    start_idx = path.index(query_nodes[i])
                    end_idx = path.index(query_nodes[i+1])
                    distance = abs(end_idx - start_idx)
                    dis = dis + distance
            else:
                round_num = 1
                for i in range(len(query_nodes)-1):
                    if round_num == 1:
                        start_idx = path.index(query_nodes[i])
                        end_idx = path.index(query_nodes[i+1])
                        distance = abs(end_idx - start_idx)
                        dis = dis + distance
                        round_num+=1
                    else:
                        start_idx = path.index(query_nodes[i])
                        end_idx = path.index(query_nodes[i+1])
                        distance = abs(end_idx - start_idx)
                        dis = dis + distance
                        round_num+=1

            paths_avg_dis.append(dis / (len(query_nodes)-1))
        avg_dis = round(sum(paths_avg_dis)/len(paths_avg_dis),3)
        
    print("paths_avg_dis:",avg_dis)


    with open('../output/Cal_Paths_Distance/E2/Path5_GreedyDist_BFS.csv', 'a+', newline='') as f6: ###
        writer = csv.writer(f6)
        writer.writerow([result[0]["Q_ID"], avg_dis])
        f6.flush()