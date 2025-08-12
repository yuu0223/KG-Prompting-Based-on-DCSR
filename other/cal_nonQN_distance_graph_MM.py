import json
import os
import csv
import re
import random
from tqdm import tqdm
from collections import deque
from itertools import combinations
import codecs
from neo4j import GraphDatabase
import BuildDatabase
from communitysearch import FindKG

### ---- 1. build neo4j knowledge graph datasets
uri = os.getenv("neo4j_uri")
username = os.getenv("neo4j_username")
password = os.getenv("neo4j_password")
print(codecs.decode(uri, 'unicode_escape')) # 檢查用

# --- build KG 
# data_path = './data/chatdoctor5k/train.txt'
# BuildDatabase.BuildNeo4j(data_path)
# ---

driver = GraphDatabase.driver(codecs.decode(uri, 'unicode_escape'), auth=(username, password))
# session = driver.session()
### --- End of Step 1

### --- new file
with open('./output/Cal_Paths_Distance/Non-QN-subgraph/Graph.csv', 'w', newline='') as f4: ###
    writer = csv.writer(f4)
    writer.writerow(['Q_ID','Distance'])


### --- Start!
# BFS 計算兩點最短距離
def bfs_distance(graph, start, end):
    queue = deque([(start, 0)])
    visited = set([start])
    
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')


with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
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
    query_nodes = result[0]["query_nodes"]
    subgraph = FindKG.find_whole_KG(driver)

    ### --- 提取 path entities
    exclude_relations = {"has_symptom", "can_check_disease", "possible_disease", "need_medical_test", "possible_cure_disease", "need_medication"}
    entity_paths = []
    for path in paths_list_path:
        ### --- for mindmap
        parts = path.split('->')
        entities = [e for e in parts if e not in exclude_relations]
        entity_paths.append(entities)
        
    print("entity_paths", entity_paths)

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
    print("非QNs節點:", non_qns_nodes)
    pairs = list(combinations(non_qns_nodes, 2))
    print("非QNs節點間的所有組合:", pairs, len(pairs))

    # 計算所有非QNs節點間的距離
    if len(pairs) == 0:
        avg_distance = 0

    else:
        total = 0
        for u, v in pairs:
            d = bfs_distance(subgraph, u, v)
            print(f"{u} <-> {v} 的距離為 {d}")
            total += d
        avg_distance = total / len(pairs)

    print(f"\n平均距離：{avg_distance}")
    

    with open('./output/Cal_Paths_Distance/Non-QN-subgraph/Graph.csv', 'a+', newline='') as f6: ###
        writer = csv.writer(f6)
        writer.writerow([result[0]["Q_ID"], round(avg_distance,3)])
        f6.flush()


