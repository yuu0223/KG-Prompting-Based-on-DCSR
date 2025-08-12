### --- def 說明
# create_graph: 建立graph。
# steiner_tree_plt: Steiner Tree 畫圖用。
# get_query_nodes_path_nodes: 依照 query nodes 順序來尋找路徑 (有overlap)。
# get_shortest_path_nodes: 不管 key entities 順序，依照 shortest paths 來尋找路徑 (無overlap)。
# get_relations_with_nodes: 將前面 method 1 or 2 找出來的 nodes，依序找出他們之間的 relations。
# join_all_path: 將所有 paths 使用 "->" 連接起來，有可能不止一條。
### ---

import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

def create_graph(graph_dict):
    # 創建無向圖
    G = nx.Graph()

    # 從無權重的字典資料中添加邊，並設置權重=1
    for node, neighbors in graph_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor, weight=1)
    
    return G

### --- Steiner Tree 畫圖用
def steiner_tree_plt(original_G, G_steiner_tree, index, match_kg):
    # 畫出原始圖
    pos = nx.spring_layout(original_G)
    # pos = nx.spring_layout(original_G, k=2, scale=2)
    # node_colors = ['lightblue' if node not in (match_kg or []) else 'orange' for node in original_G.nodes]
    # node_sizes = [300 if node not in (match_kg or []) else 600 for node in original_G.nodes]
    # nx.draw(original_G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, edge_color='gray')
    # nx.draw_networkx_labels(original_G, pos, font_size=3)
    nx.draw(original_G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    nx.draw_networkx_edges(G_steiner_tree, pos, edge_color='red', width=2)
    plt.savefig(f"steiner_tree{index}.png")
### ---

### --- 1. 尋找 paths 的 nodes
### --- 1. Method 1: 依照 query nodes 順序來尋找路徑 (有overlap)
def get_query_nodes_path_nodes(G_steiner_tree, match_kg):
    all_path_list=[]
    # 為預防query nodes之間沒有連接的path而設立的
    current_path_list=[]
    for i in range(len(match_kg)-1):
        start_node = match_kg[i]
        end_node = match_kg[i+1]

        try:
            current_path = nx.shortest_path(G_steiner_tree, source=start_node, target=end_node)
            if i > 0:
                current_path = current_path[1:]
            current_path_list.extend(current_path)
        
        except nx.shortest_path:
            all_path_list.append(current_path_list)
            print(f"No path found from {start_node} to {end_node}.")
            current_path_list=[]
        
        if i == len(match_kg)-2:
            all_path_list.append(current_path_list)
    # print("pahts:\n", "->".join(all_path_list[0]),'\n')
    return all_path_list
### --- End of Method 1


### --- 1. Method 2: 不管 key entities 順序，依照 shortest paths 來尋找路徑 (無overlap)
def get_shortest_path_nodes(original_G, G_steiner_tree, match_kg):
    all_path_list = []
    # 計算完整的路徑序列
    path_list = []
    print(G_steiner_tree.edges(data=True))

    node_counts = Counter()
    for edge in G_steiner_tree.edges():
        node_counts[edge[0]] += 1
        node_counts[edge[1]] += 1
        # 找出所有edges (E.g. (N1, N2, weighted=1))
        path = nx.shortest_path(original_G, edge[0], edge[1], weight='weight')
        path_list.append(path)
    

    while True:
        # 找到出現次數最少的節點
        min_count = min([count for count in node_counts.values() if count > 0])
        start_node = [node for node, count in node_counts.items() if count == min_count][0]

        path_join = [start_node]
        visited = set([node for node, count in node_counts.items() if count == 0])
        current_node = start_node
        while True:
            # 找到與當前節點相連的節點
            neighbors = [n2 if n1 == current_node else n1 for n1, n2 in path_list if current_node in [n1, n2]]
            # 過濾掉已訪問過的節點
            unvisited_neighbors = [node for node in neighbors if node not in visited and node not in path_join]
            print("current:",current_node)
            print(node_counts)
    
            if unvisited_neighbors == []:
                break
            else:
                node_counts[current_node] -=1
                node_counts[unvisited_neighbors[0]] -=1
                path_join.append(unvisited_neighbors[0])
                current_node = unvisited_neighbors[0]
                visited = set([node for node, count in node_counts.items() if count == 0])

        all_path_list.append(path_join)
        if all(count == 0 for count in node_counts.values()):
            break
    
    return all_path_list
### --- End of Method 2


### --- 2. 將前面 method 1 or 2 找出來的 nodes，依序找出他們之間的 relations
def get_relations_with_nodes(all_nodes_list, driver):
    all_path_list = []
    for current_list in all_nodes_list:
        paths = []
        for i in range(len(current_list)-1):
            start_node = current_list[i]
            end_node = current_list[i+1]
            query=f"""MATCH p=(n1:Entity {{name: '{start_node}'}})-[r]->(n2:Entity {{name: '{end_node}'}})
                      RETURN p"""

            with driver.session() as session:
                result = session.run(query)
                for record in result:
                    path = record['p']
                    paths.append(path.nodes[0]["name"])
                    paths.append(path.relationships[0].type)
        paths.append(current_list[-1])
        all_path_list.append(paths)
    # print("all:", all_path_list)
    return all_path_list
### ---

### --- 3. 將所有 paths 使用 "->" 連接起來，有可能不止一條
def join_all_path(all_path_list):
    if len(all_path_list) > 1:
        temp_lists=[]
        for list in all_path_list:
            temp_lists.append("->".join(list))
        final_result = "\n\n\n".join(temp_lists)
    else:
        final_result = "->".join(all_path_list[0])
    print('\n', final_result)
    return final_result
### ---