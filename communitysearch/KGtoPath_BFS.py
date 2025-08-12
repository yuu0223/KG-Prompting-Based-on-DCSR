import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from communitysearch import FindKG

### --- 優化版本 1: 減少重複的路徑查詢 (BFS版本)
def paths_in_neo4j_optimized_bfs(all_paths, top_n, driver):
    """優化版本：批量查詢和預先計算 (BFS版本)"""
    community_search_paths = {}
    
    # 預先收集所有需要查詢的路徑
    all_queries = []
    query_mapping = {}
    
    for key, value in all_paths.items():
        if key == "one_node":
            entity_name = all_paths["one_node"][0]
            query = f"MATCH (e:Entity)-[r]->(n) WHERE e.name = '{entity_name}' RETURN type(r) AS relationship_type, collect(n.name) AS neighbor_entities"
            all_queries.append((key, query, "single_node"))
            query_mapping[len(all_queries)-1] = key
        else:
            for num in range(len(value)):
                path_list = value[num]
                query = build_path_query(path_list)
                all_queries.append((key, query, "path", num, path_list))
                query_mapping[len(all_queries)-1] = key
    
    # 批量執行查詢
    results = batch_execute_queries(all_queries, driver)
    
    # 處理結果
    temp_paths_dict = {}
    for i, result in enumerate(results):
        query_info = all_queries[i]
        key = query_info[0]
        
        if query_info[2] == "single_node":
            community_search_paths["query_node"] = all_paths["one_node"]
            for record in result:
                rel_type = record["relationship_type"]
                entity = record["neighbor_entities"]
                community_search_paths[rel_type] = entity
            flag = "one_node"
        else:
            if key not in temp_paths_dict:
                temp_paths_dict[key] = []
            
            path_list = query_info[4]
            paths = []
            for record in result:
                if record['p']:
                    path = record['p']
                    for j in range(len(path_list) - 1):
                        paths.append(path.nodes[j]["name"])
                        paths.append(path.relationships[j].type)
                    paths.append(path.nodes[len(path_list) - 1]["name"])
            
            if paths:
                temp_paths_dict[key].append(paths)
    
    # 使用路徑長度篩選最短路徑（BFS特性）
    for key, temp_paths in temp_paths_dict.items():
        # 過濾空路徑
        temp_paths = [path for path in temp_paths if path]
        
        if len(temp_paths) > top_n:
            community_search_paths[key] = select_top_paths_by_length(temp_paths, top_n)
        else:
            community_search_paths[key] = temp_paths
        flag = "more_than_one_node"
    
    return community_search_paths, flag


def build_path_query(path_list):
    """建構路徑查詢語句"""
    match_statements = []
    for j in range(len(path_list) - 1):
        match_statements.append(f'(n{j+1}:Entity {{name: "{path_list[j]}"}})-[r{j+1}]->')
    match_statements.append(f'(n{len(path_list)}:Entity {{name: "{path_list[-1]}"}})')
    match_clause = "".join(match_statements)
    return f"MATCH p={match_clause}\nRETURN p"


def batch_execute_queries(queries, driver, max_workers=5):
    """批量執行查詢以減少資料庫連線開銷"""
    results = [None] * len(queries)
    
    def execute_query(index_query_pair):
        index, query_info = index_query_pair
        query = query_info[1]
        try:
            with driver.session() as session:
                result = session.run(query)
                return index, list(result)
        except Exception as e:
            print(f"查詢錯誤: {e}")
            return index, []
    
    # 使用線程池並行執行查詢
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(execute_query, (i, query)): i 
            for i, query in enumerate(queries)
        }
        
        for future in as_completed(future_to_index):
            index, result = future.result()
            results[index] = result
    
    return results


def select_top_paths_by_length(temp_paths, top_n):
    """根據路徑長度選擇最短的路徑（BFS特性）"""
    if not temp_paths:
        return []
    
    # 按照路徑長度排序，選擇最短的路徑
    temp_paths_sorted = sorted(temp_paths, key=len)[:top_n]
    return temp_paths_sorted


### --- 優化版本 2: 路徑長度快取
class PathLengthCache:
    def __init__(self):
        self.cache = {}
    
    def get_shortest_paths(self, graph_dict, source, target):
        """獲取最短路徑，使用快取避免重複計算"""
        cache_key = (source, target)
        
        if cache_key not in self.cache:
            try:
                G_subgraph = nx.DiGraph()
                for node, neighbors in graph_dict.items():
                    for neighbor in neighbors:
                        G_subgraph.add_edge(node, neighbor)
                
                # 使用BFS找到所有最短路徑
                if nx.has_path(G_subgraph, source, target):
                    shortest_paths = list(nx.all_shortest_paths(G_subgraph, source, target))
                    self.cache[cache_key] = shortest_paths
                else:
                    self.cache[cache_key] = []
            except:
                self.cache[cache_key] = []
        
        return self.cache[cache_key]


### --- 優化版本 3: 改進的主要執行流程 (BFS版本)
def optimized_community_search_workflow_bfs(result_subgraph, match_kg, driver, top_n=10):
    """優化後的完整工作流程 (BFS版本)"""
    
    # 使用快取的路徑計算
    path_cache = PathLengthCache()
    
    # 路徑查找
    all_paths = subgraph_path_finding(result_subgraph, match_kg)
    
    # 優化後的路徑處理
    path_list, flag = paths_in_neo4j_optimized_bfs(all_paths, top_n, driver)
    
    return path_list, flag


### --- 進一步優化：使用記憶化快取常見查詢
from functools import lru_cache
import hashlib

class QueryCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_query_hash(self, query):
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query):
        query_hash = self.get_query_hash(query)
        if query_hash in self.cache:
            self.access_count[query_hash] = self.access_count.get(query_hash, 0) + 1
            return self.cache[query_hash]
        return None
    
    def set(self, query, result):
        query_hash = self.get_query_hash(query)
        
        # 如果快取滿了，移除最少使用的項目
        if len(self.cache) >= self.max_size:
            least_used = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_used]
            del self.access_count[least_used]
        
        self.cache[query_hash] = result
        self.access_count[query_hash] = 1


# 全域查詢快取
query_cache = QueryCache()

def cached_query_execution(query, driver):
    """帶快取的查詢執行"""
    cached_result = query_cache.get(query)
    if cached_result is not None:
        return cached_result
    
    with driver.session() as session:
        result = list(session.run(query))
        query_cache.set(query, result)
        return result


### --- 使用快取的批量查詢執行
def batch_execute_queries_with_cache(queries, driver, max_workers=5):
    """帶快取的批量執行查詢"""
    results = [None] * len(queries)
    
    def execute_query_with_cache(index_query_pair):
        index, query_info = index_query_pair
        query = query_info[1]
        try:
            result = cached_query_execution(query, driver)
            return index, result
        except Exception as e:
            print(f"查詢錯誤: {e}")
            return index, []
    
    # 使用線程池並行執行查詢
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(execute_query_with_cache, (i, query)): i 
            for i, query in enumerate(queries)
        }
        
        for future in as_completed(future_to_index):
            index, result = future.result()
            results[index] = result
    
    return results


### --- 完整的優化BFS版本主函數
def paths_in_neo4j_optimized_bfs_full(all_paths, top_n, driver):
    """完整優化版本：批量查詢、快取和BFS路徑選擇"""
    community_search_paths = {}
    
    # 預先收集所有需要查詢的路徑
    all_queries = []
    
    for key, value in all_paths.items():
        if key == "one_node":
            entity_name = all_paths["one_node"][0]
            query = f"MATCH (e:Entity)-[r]->(n) WHERE e.name = '{entity_name}' RETURN type(r) AS relationship_type, collect(n.name) AS neighbor_entities"
            all_queries.append((key, query, "single_node"))
        else:
            for num in range(len(value)):
                path_list = value[num]
                query = build_path_query(path_list)
                all_queries.append((key, query, "path", num, path_list))
    
    # 使用快取的批量執行查詢
    results = batch_execute_queries_with_cache(all_queries, driver)
    
    # 處理結果
    temp_paths_dict = {}
    for i, result in enumerate(results):
        query_info = all_queries[i]
        key = query_info[0]
        
        if query_info[2] == "single_node":
            community_search_paths["query_node"] = all_paths["one_node"]
            for record in result:
                rel_type = record["relationship_type"]
                entity = record["neighbor_entities"]
                community_search_paths[rel_type] = entity
            flag = "one_node"
        else:
            if key not in temp_paths_dict:
                temp_paths_dict[key] = []
            
            path_list = query_info[4]
            paths = []
            for record in result:
                if record['p']:
                    path = record['p']
                    for j in range(len(path_list) - 1):
                        paths.append(path.nodes[j]["name"])
                        paths.append(path.relationships[j].type)
                    paths.append(path.nodes[len(path_list) - 1]["name"])
            
            if paths:
                temp_paths_dict[key].append(paths)
    
    # 使用路徑長度篩選最短路徑
    for key, temp_paths in temp_paths_dict.items():
        # 過濾空路徑
        temp_paths = [path for path in temp_paths if path]
        
        if len(temp_paths) > top_n:
            community_search_paths[key] = select_top_paths_by_length(temp_paths, top_n)
        else:
            community_search_paths[key] = temp_paths
        flag = "more_than_one_node"
    
    return community_search_paths, flag


### --- 原始函數保持不變（供參考）
def subgraph_path_finding(result_subgraph, match_kg):
    """原始的路徑查找函數（保持不變）"""
    all_paths = {}
    if len(match_kg) > 1:
        query_node_set = []
        [query_node_set.append([match_kg[i], match_kg[i+1]]) for i in range(len(match_kg)-1)]

        for start_n, end_n in query_node_set:
            all_paths[(start_n, end_n)] = FindKG.subgraph_BFS(result_subgraph, start_n, end_n)
    else:
        all_paths["one_node"] = match_kg
    
    return all_paths