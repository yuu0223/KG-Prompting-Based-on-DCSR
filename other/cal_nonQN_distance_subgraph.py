import json
import os
import csv
import re
import random
from tqdm import tqdm
from collections import deque
from itertools import combinations

### --- new file
with open('../output/Cal_Paths_Distance/Non-QN-subgraph/GreedyGen_BFS.csv', 'w', newline='') as f4: ###
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


with open("../data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

with open("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/Final_Q714_Gemini1.5/GreedyGen/20250505_BFS_Gemini15_2.json", "r", encoding='utf-8') as f1: ###
    data = json.load(f1)

round_count = 0
# skip_id = [2574, 4405, 938, 4409]

for line in tqdm(lines[round_count+429:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
    x = json.loads(line)

    id = x["Q_ID"]
    result = [item for item in data if item.get("Q_ID") == id]
    print("Q_ID from paths json:", result[0]["Q_ID"])
    # print(result)

    paths_list = result[0]["paths-list"] # list
    query_nodes = result[0]["query_nodes"]
    subgraph = result[0]["subgraph_dict"]

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
    

    with open('../output/Cal_Paths_Distance/Non-QN-subgraph/GreedyGen_BFS.csv', 'a+', newline='') as f6: ###
        writer = csv.writer(f6)
        writer.writerow([result[0]["Q_ID"], round(avg_distance,3)])
        f6.flush()


