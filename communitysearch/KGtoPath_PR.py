import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from communitysearch import FindKG

### --- 優化版本 1: 減少重複的路徑查詢
def paths_in_neo4j_optimized(all_paths, pagerank_values, top_n, driver):
    """優化版本：批量查詢和預先計算"""
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
    
    # 使用 PageRank 篩選最佳路徑
    for key, temp_paths in temp_paths_dict.items():
        if len(temp_paths) > top_n:
            community_search_paths[key] = select_top_paths_by_pagerank(temp_paths, pagerank_values, top_n)
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


def select_top_paths_by_pagerank(temp_paths, pagerank_values, top_n):
    """使用向量化計算快速選擇 PageRank 分數最高的路徑"""
    if not temp_paths:
        return []
    
    avg_scores = []
    for path in temp_paths:
        # 只取節點（偶數索引位置）
        nodes = [path[i] for i in range(0, len(path), 2)]
        # 計算平均 PageRank 值
        scores = [pagerank_values.get(node, 0) for node in nodes]
        avg_scores.append(np.mean(scores) if scores else 0)
    
    # 獲取分數最高的前 N 個索引
    top_indices = np.argsort(avg_scores)[-top_n:][::-1]
    return [temp_paths[i] for i in top_indices]


### --- 優化版本 2: 預先計算 PageRank 並快取
class PageRankCache:
    def __init__(self):
        self.cache = {}
        self.last_graph_hash = None
    
    def get_pagerank(self, graph_dict):
        """獲取 PageRank 值，使用快取避免重複計算"""
        # 簡單的圖雜湊值計算
        graph_hash = hash(str(sorted(graph_dict.items())))
        
        if graph_hash != self.last_graph_hash or not self.cache:
            G_subgraph = nx.DiGraph()
            for node, neighbors in graph_dict.items():
                for neighbor in neighbors:
                    G_subgraph.add_edge(node, neighbor)
            
            self.cache = nx.pagerank(G_subgraph, alpha=0.85)
            self.last_graph_hash = graph_hash
        
        return self.cache


### --- 優化版本 3: 改進的主要執行流程
def optimized_community_search_workflow(graph_dict, result_subgraph, match_kg, driver, top_n=10):
    """優化後的完整工作流程"""
    
    # 使用快取的 PageRank 計算
    pagerank_cache = PageRankCache()
    pagerank_values = pagerank_cache.get_pagerank(graph_dict)
    
    # 路徑查找
    all_paths = subgraph_path_finding(result_subgraph, match_kg)
    
    # 優化後的路徑處理
    path_list, flag = paths_in_neo4j_optimized(all_paths, pagerank_values, top_n, driver)
    
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