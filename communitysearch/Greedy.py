### --- def 說明
# greedy_DFS: 使用DFS來遍歷圖形，並標記已訪問的節點。 [greedy_max_min_degree內的功能]
# find_node: 判斷quey nodes各自可以走訪哪些nodes，透過DFS來尋找。 [greedy_max_min_degree內的功能]
# check_same_subgraph: 檢查 Query Nodes 是否都有在同一個 subgraph 內 (相互連接)。 [greedy_max_min_degree內的功能]
# greedy_max_min_degree: Community Search without size restriction。
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
    if len(query_nodes) > 1:
        for node in query_nodes:
            visited = set()
            output = greedy_DFS(graph, node, visited)
            query_node_visited[node] = output

    elif len(query_nodes) == 1:
        visited = set()
        visit = greedy_DFS(graph, query_nodes[0], visited)
        query_node_visited[query_nodes[0]] = visit
    
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



### --- Community Search without size restriction (單純刪除 one minimum degree)
def greedy_max_min_degree(graph, query_nodes):
    """
    Implementation of the Greedy algorithm for maximizing the minimum degree.

    Args:
        graph (dict): Medical Knowledge Graph。
        query_nodes (set): 查詢節點的集合。

    Returns:
        dict: Maximize minimum degree subgraph。
    """
    # 初始化 G0 為輸入圖
    G = graph.copy()

    while True:
        # 在 G 中找到度數最小的節點
        min_degree_node = min(G, key=lambda node: len(G[node]))

        G_next = G.copy()
        
        ### 檢查是否query nodes是否為minimum degree
        for node in query_nodes:
            if node not in G:
                return f"There's no query node name like {node}." #當遇到Graph裡面沒有的query nodes
            else:
                if (len(G[node]) == len(G[min_degree_node])):
                    return G

        ### 從 G 中刪除具有最小度數的節點及其相關邊
        del G_next[min_degree_node]
        for node in G_next:
            G_next[node] = [neighbor for neighbor in G_next[node] if neighbor != min_degree_node]
        ###### 判斷quey nodes各自可以走訪哪些nodes
        check = check_same_subgraph(query_nodes, find_node(G_next, query_nodes))
        if (check == 0):
            G = G_next
        elif (check == 1):
            break
        else:
            return "No Query Nodes."
    
    return G
### ---


### ---  Example1
# graph = {
#     'a': ['b', 'c', 'd'],
#     'b': ['a', 'c', 'd', 'e'],
#     'c': ['a', 'b', 'd', 'e', 'f'],
#     'd': ['a', 'b', 'c', 'e', 'f'],
#     'e': ['b', 'c', 'd', 'f'],
#     'f': ['c', 'd', 'e']
# }
# query_nodes = {'a', 'b', 'c'}

### ---  Example2
### Answer: Delete nodeC (mi).
# graph = {
#     'a': ['c', 'd', 'f','e'],
#     'b': ['c', 'd'],
#     'c': ['a', 'b', 'd', 'e'],
#     'd': ['a','b', 'c', 'e'],
#     'e': ['a', 'c', 'd', 'f'],
#     'f': ['a', 'e']
# }
# query_nodes = ['a', 'd']

### ---  Example3
# graph = {
#     'a': ['c', 'd', 'e'],
#     'b': [],
#     'c': ['a', 'd'],
#     'd': ['a', 'c'],
#     'e': ['a']
# }
# query_nodes = ['e','c']


# result_subgraph = greedy_max_min_degree(graph, query_nodes)
# print(result_subgraph)