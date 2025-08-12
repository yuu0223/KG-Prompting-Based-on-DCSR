### --- def 說明
# find_whole_KG: 將整個 KG 存成 dict。
# single_way_edge_finding: 建立單向 paths。
# combine_lists_mindmap: 將所有兩兩節點間的paths做排列組合，並合併起來成一條完整的path。 [double_way_edge_finding內的功能]
# double_way_edge_finding: 建立雙向 paths。
# subgraph_DFS: 在尋找 paths 的過程使用 DFS。 [subgraph_path_finding內的功能]
# subgraph_BFS: 在尋找 paths 的過程使用 BFS。 [subgraph_path_finding內的功能，目前使用這個!]
# subgraph_path_finding: 執行完 Community Search 後，依照 key entities 順序來尋找 paths 的 nodes。
# paths_in_neo4j:找出「兩兩節點間」完整的路徑(包含nodes, edges)，並選取字數最短的 top N 條 (這邊預設N=10)。
# paths_in_neo4j_for_PageRank: 找出「兩兩節點間」完整的路徑(包含nodes, edges)，並計算每條路徑的平均PR值，選取分順最高的 top N 條 (這邊預設N=10)。
# combine_lists: 找出最終「整條完整的path」，將 path_in_Neo4j 找出來的結果排列組合 & 選取字數最短的 top N 條 (預設N=10)。
### ---

from communitysearch import PromptGenerate
import itertools
import random
random.seed(42)

def find_whole_KG(driver):
    graph_dict = {}
    session = driver.session()

    # 檢索所有節點及其相鄰節點
    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN n, collect(m) as neighbors
    """
    result = session.run(query)

    # 將查詢結果轉換為字典形式
    for record in result:
        node = record['n']
        neighbors = record['neighbors']
        graph_dict[node['name']] = [neighbor['name'] for neighbor in neighbors]
    
    return graph_dict


def single_way_edge_finding(result_subgraph, driver):
    path_list, sure_find = [], []
    for current_node in list(result_subgraph.keys()):
        if current_node not in sure_find:
            for candidate_node in result_subgraph[current_node]:
                if candidate_node not in sure_find:
                    path = PromptGenerate.find_select_node_path(current_node, candidate_node, driver)
                    path_list.append(path)
                else:
                    continue

            sure_find.append(current_node)
        else:
            continue

    return path_list


def combine_lists_mindmap(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results


def double_way_edge_finding(result_subgraph, driver):
    community_result_path = []
    for start_node in list(result_subgraph.keys()):
        for end_node in result_subgraph[start_node]:
            path = PromptGenerate.find_select_node_path(start_node, end_node, driver)
            community_result_path.append(path)
    
    print('twoway_path: ', len(community_result_path))

    path_list = []
    for p in community_result_path:
        path_list.append(p.split('->'))

    community_result_path = combine_lists_mindmap(*path_list)

    return community_result_path



### ---- Path Finding - DFS
def subgraph_DFS(result_subgraph, start_n, end_n, path=None):

    path = [start_n] if path is None else path + [start_n]
        
    if start_n == end_n:
        return [path]
    
    paths = []
    for node in result_subgraph[start_n]:
        if node not in path:
            new_paths = subgraph_DFS(result_subgraph, node, end_n, path)
            for p in new_paths:
                if len(p) <= 6:
                    paths.append(p)
                
    return paths
### ---


### --- Path Finding - BFS
def subgraph_BFS(result_subgraph, start_n, end_n):
    from collections import deque, defaultdict

    # Step 1: Run BFS to find the shortest distance to each node
    queue = deque([start_n])
    distances = {start_n: 0}
    parents = defaultdict(list)  # stores all possible parents of each node
    
    while queue:
        node = queue.popleft()
        
        for neighbor in result_subgraph[node]:
            if neighbor not in distances:  # first time visiting this node
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
            
            if distances[neighbor] == distances[node] + 1:  # valid parent for shortest path
                parents[neighbor].append(node)
    
    # Step 2: Backtrack to find all shortest paths
    def backtrack(current, path):
        if current == start_n:
            paths.append(path[::-1])  # store the path
            return
        for parent in parents[current]:
            backtrack(parent, path + [parent])

    # Start backtracking from the target node
    paths = []
    backtrack(end_n, [end_n])
    
    return paths
### ---

### --- 執行完 Community Search 後，依照 key entities 順序來尋找 paths 的 nodes
def subgraph_path_finding(result_subgraph, match_kg):
    ### 每組要找path的start node和end node (e.g. [['a', 'b'], ['b', 'c']])
    all_paths = {}
    if len(match_kg) > 1:
        query_node_set = []
        [query_node_set.append([match_kg[i], match_kg[i+1]]) for i in range(len(match_kg)-1)]

        for start_n, end_n in query_node_set:
            all_paths[(start_n, end_n)] = subgraph_BFS(result_subgraph, start_n, end_n)

    else:
        all_paths["one_node"] = match_kg

    
    # ### 多一個篩選：兩兩query nodes之間距離不超過5-hops
    # all_paths_5hops = {}
    # for start_n, end_n in query_node_set:
    #     # 分別檢查兩兩 query nodes 之間的各個 paths 是否超過 5-hops
    #     sub_path_list = []
    #     for i in range(len(all_paths[(start_n, end_n)])):
    #         current_check_path = all_paths[(start_n, end_n)][i]
    #         if len(current_check_path) <= 6:
    #             sub_path_list.append(current_check_path)
        
    #     all_paths_5hops[(start_n, end_n)] = sub_path_list
    # print("all:", all_paths)
    return all_paths
    ### e.g. all_paths = {('A', 'D'): [['A', 'D']], ('D', 'C'): [['D', 'A', 'B', 'C']]}
### ---


### --- 找出完整的路徑(包含nodes, edges)，並選取字數最短的 top N 條 (這邊預設N=10)
# def paths_in_neo4j(all_paths, top_n, driver):
#     community_search_paths = {} ###topn
#     # community_search_paths = [] ###top1
#     for key, value in all_paths.items():
#         if key == "one_node":
#             entity_name = all_paths["one_node"][0]
#             query = f"MATCH (e:Entity)-[r]->(n) WHERE e.name = '{entity_name}' RETURN type(r) AS relationship_type, collect(n.name) AS neighbor_entities"
#             session = driver.session()
#             result = session.run(query)

#             community_search_paths["query_node"] = all_paths["one_node"]
#             for record in result:
#                 rel_type = record["relationship_type"]
#                 entity = record["neighbor_entities"]
#                 community_search_paths[rel_type] = entity
        
#             flag = "one_node"

#         else:
#             temp_paths = [] #用來存放start_n, end_n相同的path
#             for num in range(len(value)):
#                 path_list = value[num]
#                 match_statements = []
#                 for j in range(len(path_list) - 1):
#                     match_statements.append(f'(n{j+1}:Entity {{name: "{path_list[j]}"}})-[r{j+1}]->')
#                 # 最后的节点不需要连接符
#                 match_statements.append(f'(n{len(path_list)}:Entity {{name: "{path_list[-1]}"}})')
#                 # 将所有MATCH部分拼接起来
#                 match_clause = "".join(match_statements)
#                 query = f"MATCH p={match_clause}\nRETURN p"

#                 with driver.session() as session:
#                     query = f"""{query}"""
#                     result = session.run(query)
#                     paths = []
#                     for record in result:
#                         path = record['p']
#                         for j in range(len(path_list) - 1):
#                             paths.append(path.nodes[j]["name"])
#                             paths.append(path.relationships[j].type)
#                         paths.append(path.nodes[len(path_list) - 1]["name"])
                        
#                     temp_paths.append(paths)
                
#                 temp_paths = [path for path in temp_paths if path]
#                 if len(temp_paths) > top_n:
#                     temp_paths_sorted = sorted(temp_paths, key=len)[:top_n]
#                     community_search_paths[key] = temp_paths_sorted
#                 else:
#                     community_search_paths[key] = temp_paths    

#             flag = "more_than_one_node"             
        
#         ### --- 原本只取一條用的
#         # if len(temp_paths) > 0:
#         #     #每回合的paths中隨機選取一條path
#         #     # chosen_path = random.choice(temp_paths)
#         #     chosen_path = temp_paths[0]
#         #     # community_search_paths[key] = chosen_path  ### 每回合所有paths儲存用
#         #     # community_search_paths.append(chosen_path) ### 每回合單條path儲存用
#         #     if community_search_paths == []:
#         #         community_search_paths.append(chosen_path)
#         #     else:
#         #         if chosen_path[0] == community_search_paths[path_num][-1]:
#         #             community_search_paths[path_num]+=chosen_path[1:]
#         #         else:
#         #             community_search_paths.append(chosen_path)
#         #             path_num+=1
#         ### ----
#     # print(community_search_paths)
        
#     return community_search_paths, flag
### ---


# ### --- 找出完整的路徑(包含nodes, edges)，並計算每條路徑的平均PR值，選取分順最高的 top N 條 (這邊預設N=10)
# def paths_in_neo4j_for_PageRank(all_paths, pagerank_values, top_n, driver):
#     community_search_paths = {} ###top5
#     for key, value in all_paths.items():
#         if key == "one_node":
#             entity_name = all_paths["one_node"][0]
#             query = f"MATCH (e:Entity)-[r]->(n) WHERE e.name = '{entity_name}' RETURN type(r) AS relationship_type, collect(n.name) AS neighbor_entities"
#             session = driver.session()
#             result = session.run(query)

#             community_search_paths["query_node"] = all_paths["one_node"]
#             for record in result:
#                 rel_type = record["relationship_type"]
#                 entity = record["neighbor_entities"]
#                 community_search_paths[rel_type] = entity
        
#             flag = "one_node"

#         else:
#             temp_paths = [] #用來存放start_n, end_n相同的path
#             for num in range(len(value)):
#                 path_list = value[num]
#                 match_statements = []
#                 for j in range(len(path_list) - 1):
#                     match_statements.append(f'(n{j+1}:Entity {{name: "{path_list[j]}"}})-[r{j+1}]->')
#                 # 最後一個節點不需要連接符號
#                 match_statements.append(f'(n{len(path_list)}:Entity {{name: "{path_list[-1]}"}})')
#                 # 把所有 query 整合起來
#                 match_clause = "".join(match_statements)
#                 query = f"MATCH p={match_clause}\nRETURN p"

#                 with driver.session() as session:
#                     query = f"""{query}"""
#                     result = session.run(query)
#                     paths = []
#                     for record in result:
#                         path = record['p']
#                         for j in range(len(path_list) - 1):
#                             paths.append(path.nodes[j]["name"])
#                             paths.append(path.relationships[j].type)
#                         paths.append(path.nodes[len(path_list) - 1]["name"])
#                     temp_paths.append(paths)
                
#                 temp_paths = [path for path in temp_paths if path]
#                 if len(temp_paths) > top_n:
#                     # 計算各條paths的平均PageRank值
#                     avg_PR = []
#                     for list in temp_paths:
#                         count=0
#                         sum_PR=0
#                         for i in range(0,len(list),2):
#                             sum_PR+=pagerank_values[list[i]]
#                             count+=1
#                         avg_PR.append(sum_PR/count)

#                     # 選取平均分數最高的前 N 條 paths
#                     top_n_index = sorted(range(len(avg_PR)), key=lambda i: avg_PR[i], reverse=True)[:top_n]
#                     community_search_paths[key] = [temp_paths[index] for index in top_n_index]
#                 else:
#                     community_search_paths[key] = temp_paths 

#             flag = "more_than_one_node"              
        
#     return community_search_paths, flag
# ### ---


### --- 排列組合 & 選取字數最短的 top N 條 (預設N=10)
def combine_lists(community_search_paths, pagerank_values, top_n, flag):
    if flag == "one_node":
        print("test")
        query_node = community_search_paths["query_node"][0]
        final_result = []
        final_result_list = []
        pre_str = f"{query_node}->"
        path_nodes_count = 1
        for key, value in community_search_paths.items():
            if key != "query_node":
                # join每條paths by "->"
                final_result.append(pre_str+key+"->"+", ".join(value))
                path_nodes_count+=len(value)

                # neighbor分開(繪圖用)
                for item in value:
                    final_result_list.append(pre_str+key+"->"+item)

        final_result = "\n\n\n".join(final_result)
   
    elif flag == "more_than_one_node":
        # 取得所有的排列組合
        combinations = itertools.product(*community_search_paths.values())

        # 將結果串起來
        result = []
        for combination in combinations:
            combined_list = combination[0][:]  # 從第一個list開始
            for next_list in combination[1:]:
                combined_list += next_list[1:]
            result.append(combined_list)
        
        if pagerank_values == None:
            ### --- Method1：取最短 N 條路徑(Original)
            result_n = sorted(result, key=len)[:top_n]
            # print("len_n: ", result_n)
            ### ---
        else:
            ### --- Method2：PageRank Average & Maximum PR
            # 計算各條paths的平均PageRank值
            avg_PR = []
            for list in result:
                count=0
                sum_PR=0
                for i in range(0,len(list),2):
                    sum_PR+=pagerank_values[list[i]]
                    count+=1
                avg_PR.append(sum_PR/count)

            #選取平均分數最高的前 N 條paths
            top_n_index = sorted(range(len(avg_PR)), key=lambda i: avg_PR[i], reverse=True)[:top_n]
            result_n = [result[index] for index in top_n_index]
            ### --- 

        ### --- 計算節點用
        path_nodes = []
        for list in result_n:
            for i in range(0, len(list), 2):
                path_nodes.append(list[i])
        path_nodes_count = len(set(path_nodes))
        ### ---

        # join每條paths by "->"
        final_result = []
        for list in result_n:
            join_list = "->".join(list)
            final_result.append(join_list)

        final_result_list = final_result
        final_result = "\n\n\n".join(final_result)

    else:
        final_result = "There's no any reference path in this case."
        final_result_list = []
        path_nodes_count = 1

    # print("join_path: \n",final_result_list)
    return final_result, final_result_list, path_nodes_count
    # return final_result
