import KG_vision_pyvis, KG_vision_test
import json

with open("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/20250505_BFS_Gemini15.json", "r") as f:
    graph_data = json.load(f)

index = 3783
result = [item for item in graph_data if item.get("Q_ID") == index]
query_nodes = result[0]["query_nodes"]
path_join_list = result[0]["paths-list"]


### --- 8. Visualization Subgraph of Community Search
# process_name = "GreedyDist_BFS"
# # 目前作法是一條 path 一張圖 (之後想改成多條 paths 一張圖!)
# for i in range(len(path_join_list)):
#     KG_vision_pyvis.draw_subgraph_only_paths(index, path_join_list[i], query_nodes, i, process_name, position_file=None)
### --- End of Step 8


### --- pyvis test combine
# 修改你的主要調用程式碼
def visualize_combined_paths():
    """
    主要的可視化調用函數
    """
    process_name = "GreedyDist_BFS"
    
    # 假設 path_join_list 是你的路径列表
    # path_join_list = [path1, path2, path3, ...] # 你的10條路径
    
    # 調用合併可視化函數
    KG_vision_test.draw_subgraph_combined_paths(
        index,  # round_count
        path_join_list,  # 所有路径的列表
        query_nodes,  # match_kg 查詢節點
        process_name,
        position_file=None
    )
    
    print(f"已成功合併 {len(path_join_list)} 條路径到一張圖中！")


# 如果你想要同時保留原來的單個路径圖和新的合併圖，可以這樣做：
def visualize_both_individual_and_combined():
    """
    同時生成個別路径圖和合併圖
    """
    process_name = "GreedyDist_BFS"
    
    # 1. 生成個別路径圖（原來的方法）
    for i in range(len(path_join_list)):
        KG_vision_pyvis.draw_subgraph_only_paths(
            index, path_join_list[i], query_nodes, i, process_name, position_file=None
        )
    
    # 2. 生成合併路径圖（新方法）
    KG_vision_test.draw_subgraph_combined_paths(
        index, path_join_list, query_nodes, process_name, position_file=None
    )
    
    print(f"已生成 {len(path_join_list)} 張個別路径圖和 1 張合併路径圖！")



# # 🔴 新增: 修改主要調用程式碼，確保顏色一致性
# def visualize_individual_and_combined_paths():
#     """
#     生成個別路径圖和合併圖，確保顏色一致
#     """
#     process_name = "GreedyDist_BFS"
    
#     # 1. 生成個別路径圖（使用修改後的函數）
#     for i in range(len(path_join_list)):
#         KG_vision_pyvis.draw_subgraph_only_paths(
#             index, path_join_list[i], query_nodes, i, process_name, position_file=None
#         )
    
#     # 2. 生成合併路径圖
#     KG_vision_pyvis.draw_subgraph_combined_paths(
#         index, path_join_list, query_nodes, process_name, position_file=None
#     )
    
#     print(f"已生成 {len(path_join_list)} 張個別路径圖（水平排列）和 1 張合併路径圖！")
#     print("所有圖的顏色系統保持一致！")
### ---


### --- 8. Visualization Subgraph of Community Search 
process_name = "GreedyDist_BFS"

# 生成個別路径圖（原來的方式）
for i in range(len(path_join_list)): 
    KG_vision_pyvis.draw_subgraph_only_paths(index, path_join_list[i], query_nodes, i, process_name, position_file=None)

# 生成合併路径圖（新方式）
KG_vision_test.draw_subgraph_combined_paths(index, path_join_list, query_nodes, process_name, position_file=None)
### --- End of Step 8