# 計算每一題Query Nodes之間的最短距離
from neo4j import GraphDatabase
import codecs
import os
import json
import csv
from itertools import combinations

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
### --- End of Step 1


with open('../output/cal_QN_shortest_distance_new.csv', 'w', newline='') as f4: ###
    writer = csv.writer(f4)
    writer.writerow(['Q_ID','Query_Nodes','Query_Nodes_Count','Total_Distance','Average_Distance'])


def compute_all_pair_distances(node_list, session):
    """
    給定一組 query_nodes，計算所有兩兩節點間最短距離
    """
    total_distance = 0
    pair_distances = []

    for n1, n2 in combinations(node_list, 2):
        result = session.run("""
            MATCH (start:Entity {name: $n1}), (end:Entity {name: $n2})
            MATCH path = shortestPath((start)-[*..10]-(end))
            RETURN length(path) AS distance
        """, n1=n1, n2=n2)

        record = result.single()
        distance = record["distance"] if record else None

        pair_distances.append({
            "from": n1,
            "to": n2,
            "distance": distance
        })

        if distance is not None:
            total_distance += distance
    print("len:", len(pair_distances))
    return pair_distances, total_distance


output = []
with open("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/20250505_BFS_Gemini15.json", "r") as f:
    data = json.load(f)
    with driver.session() as session:
        for entry in data:
            qid = entry["Q_ID"]
            query_nodes = entry["query_nodes"]
            pair_distances, total_distance = compute_all_pair_distances(query_nodes, session)
            avg_distance = round(total_distance / len(pair_distances),3) if len(pair_distances) >= 1 else 0
            print(f"Query ID: {qid}, Pair Distances: {pair_distances}, Total Distance: {total_distance}")
            print(f"Average Distance: {round(total_distance / len(pair_distances),3) if len(pair_distances) >= 1 else 0}")

            output.append({
                "Q_ID": qid,
                "query_nodes": query_nodes,
                "pair_distances": pair_distances,
                "total_distance": total_distance,
                "avg_distance": avg_distance
            })

            with open('../output/cal_QN_shortest_distance_new.csv', 'a+', newline='') as f6:
                writer = csv.writer(f6)
                writer.writerow([qid, query_nodes, len(query_nodes), total_distance, avg_distance])
                f6.flush()

            with open("../output/cal_QN_shortest_distance_new.json", "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
