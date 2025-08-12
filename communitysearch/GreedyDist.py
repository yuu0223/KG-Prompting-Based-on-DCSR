### --- def 說明
# compute_distance_to_query_nodes_bfs: 每回合都要檢查每個 nodes 與各 query nodes 的距離是否超出限制。 [GreedyGen內的功能]
# del_minimum_node: 一次刪除一個 minimum degree node。 [GreedyGen內的功能]
# greedy_gen: GreedyGen，刪除 minimum degree node。 [GreedyDist內的功能]
# greedy_dist: GreedyDist，包含 size & distance restriction。
### ---

### --- 每回合都要檢查每個 nodes 與各 query nodes 的距離是否超出限制
def compute_distance_to_query_nodes_bfs(graph, query_nodes):
    # 將每個node的距離存在dict裡
    distances = {node: float('inf') for node in graph}
    
    # 用來計算每個node距離其中一個query node的距離變化
    stack = [(node, 0) for node in query_nodes]
    
    # 將query nodes的距離設置為0
    for node in query_nodes:
        distances[node] = 0
    
    # 開始執行BFS
    while stack:
        current_node, current_distance = stack.pop()
        
        for neighbor in graph[current_node]:
            # 更新current node的鄰居距離，並將current node pop
            if current_distance + 1 < distances[neighbor]:
                distances[neighbor] = current_distance + 1
                stack.append((neighbor, current_distance + 1))
    
    return distances
### ---


### --- DFS
def greedy_DFS(graph, start, visited):
    """
    使用深度優先搜索來遍歷圖形，並標記已訪問的節點。

    Args:
        graph (dict): Medical Knowledge Graph。
        start (str): 開始節點。
        visited (set): 已訪問的節點集合。 

    Returns:
        set: 所有與開始節點連通的節點集合。
    """
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            greedy_DFS(graph, neighbor, visited)

    return visited
### --- 
### --- 判斷quey nodes各自可以走訪哪些nodes，透過DFS來尋找
def find_node(graph, query_nodes):
    """
    判斷quey nodes各自可以走訪哪些nodes，透過DFS來尋找。

    Args:
        graph (dict): Medical Knowledge Graph。
        query nodes (list): 欲查詢節點。

    Returns:
        dict: 包含所有query nodes各自造訪的節點。
    """
    query_node_visited = dict()
    for node in query_nodes:
        visited = set()
        query_node_visited[node] = greedy_DFS(graph, node, visited)
    
    return query_node_visited
### ---
### --- 檢查 Query Nodes 是否都有在同一個 subgraph 內 (相互連接)
def check_same_subgraph(query_nodes, query_node_visited):
    """
    從DFS找出的路徑中查看query nodes有沒有在同一個subgraph內。

    Args:
        query nodes (list): 欲查詢節點。
        query_node_visited (dict): 所有query nodes各自可以造訪的節點。

    Returns:
        -1/0/1 (num): 0=所有qn都在同個graph裡 / 1=qn不在同個graph裡 / -1=input沒有qn
    """
    if query_node_visited == {}:
        return -1
    else:
        count = 0
        for key_node in query_nodes:
            for check_node in query_nodes:
                if check_node not in query_node_visited[key_node]:
                    count+=1
                    break
        return 0 if (count == 0) else 1
### ---


### --- GreedyGen
def greedy_gen(graph, query_nodes, condition_constraint):

    G = graph.copy()

    # Step1: 找出違反distance限制的所有節點，並刪除
    current_node_index = 0
    while True:
        distance = compute_distance_to_query_nodes_bfs(G, query_nodes)
        sorted_distances = dict(sorted(distance.items(), key=lambda item: item[1], reverse=True))
        current_node = list(sorted_distances.keys())[current_node_index]
        
        if sorted_distances[current_node] > condition_constraint['distance']:

            G_temp = G.copy()
            del G_temp[current_node]
            for node in G_temp:
                G_temp[node] = [neighbor for neighbor in G_temp[node] if neighbor != current_node]

            # 檢查刪除後是否連通
            check_num = check_same_subgraph(query_nodes, find_node(G_temp, query_nodes))
            # 有連通，直接刪除該節點
            if check_num == 0:
                G = G_temp
            # 沒有連通，跳下一個節點來刪除
            else:
                current_node_index+=1
        else:
            break

    
    # Step2: 沒有違反distance的節點，刪除minimum degree nodes
    while True:
        G_temp = G.copy()

        # 找出一個目前的minimum degree node
        min_degree_node = min(G_temp, key=lambda node: len(G_temp[node]))
        if min_degree_node in query_nodes:
            break
    
        # 刪除一個minimum degree node
        del G_temp[min_degree_node]
        for node in G_temp:
            G_temp[node] = [neighbor for neighbor in G_temp[node] if neighbor != min_degree_node]
        
        # 檢查是否連通
        check_num = check_same_subgraph(query_nodes, find_node(G_temp, query_nodes))
        if check_num == 0:
            G = G_temp
        else:
            break

    return G
### ---

### --- GreedyDist (Distance & Size Restriction)
def greedy_dist(graph, query_nodes, condition_constraint):
    
    G = graph.copy()
    G = greedy_gen(G, query_nodes, condition_constraint)

    graph_size = len(list(G.keys()))
    while graph_size > condition_constraint['size']:
        if condition_constraint['distance'] > 1:
            condition_constraint['distance']-=1
            G = greedy_gen(G, query_nodes, condition_constraint)
            graph_size = len(list(G.keys()))
        else:
            break
    
    # print("distance restriction:", condition_constraint['distance'])
    # return G
    return condition_constraint['distance'] ,G
### ---


### --- GreedyGen
def greedy_gen_subgraph(graph, query_nodes, condition_constraint):

    G = graph.copy()

    # Step1: 找出違反distance限制的所有節點，並刪除
    while True:
        distance = compute_distance_to_query_nodes_bfs(G, query_nodes)
        sorted_distances = dict(sorted(distance.items(), key=lambda item: item[1], reverse=True))
        current_node = list(sorted_distances.keys())[0]
        
        if sorted_distances[current_node] > condition_constraint['distance']:
            G_temp = G.copy()
            del G_temp[current_node]
            for node in G_temp:
                G_temp[node] = [neighbor for neighbor in G_temp[node] if neighbor != current_node]
            
            G = G_temp
        else:
            break
    
    # Step2: 沒有違反distance的節點，刪除minimum degree nodes
    while True:
        G_temp = G.copy()

        # 找出一個目前的minimum degree node
        min_degree_node = min(G_temp, key=lambda node: len(G_temp[node]))
        if min_degree_node in query_nodes:
            break
    
        # 刪除一個minimum degree node
        del G_temp[min_degree_node]
        for node in G_temp:
            G_temp[node] = [neighbor for neighbor in G_temp[node] if neighbor != min_degree_node]
        
        G = G_temp
        

    return G
### ---

### --- GreedyDist without check in the same subgraph (GreedyGen + check query nodes in the same subgraph)
def greedydist_subgraph(graph, query_nodes, condition_constraint):
    
    G = graph.copy()
    G = greedy_gen_subgraph(G, query_nodes, condition_constraint)

    graph_size = len(list(G.keys()))
    while graph_size > condition_constraint['size']:
        if condition_constraint['distance'] > 1:
            condition_constraint['distance']-=1
            G = greedy_gen_subgraph(G, query_nodes, condition_constraint)
            graph_size = len(list(G.keys()))
        else:
            break
    
    # print("distance restriction:", condition_constraint['distance'])
    # return G
    return condition_constraint['distance'] ,G
### ---


### --- Example 1
# graph = {
#     'A': ['B', 'C'],
#     'B': ['A', 'D', 'E'],
#     'C': ['A', 'F'],
#     'D': ['B'],
#     'E': ['B', 'F'],
#     'F': ['C', 'E']
# }
# query_nodes = ['A', 'D']

### --- Example 2
# graph = {
#     'A': ['B'],
#     'B': ['A', 'C'],
#     'C': ['B', 'D'],
#     'D': ['C', 'E'],
#     'E': ['D']
# }
# query_nodes = ['A', 'E']

### --- Example 3
# graph = {
#     'A': ['C', 'D', 'F'],
#     'B': ['C', 'D'],
#     'C': ['A', 'B'],
#     'D': ['A', 'B','E'],
#     'E': ['D'],
#     'F': ['A']
# }
# query_nodes = ['A']


# distances = compute_distance_to_query_nodes_bfs(graph, query_nodes)
# print(distances)
# condition_constraint = {'distance':1}
# G = greedydist_without_size(graph, query_nodes, condition_constraint)
# print(G)