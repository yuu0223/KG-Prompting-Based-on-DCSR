from pyvis.network import Network
import networkx as nx
from datetime import date
import os
import math
import json

PATH_FOR_PYVIS = "/home/yuu0223/KG-Prompting-Based-on-Community-Search/pyvis_result/"


def parse_paths(paths):
    """
    將文字路徑解析為一組有向邊
    :param paths: 路徑的文字表示，例如 "1->2->3->4"
    :return: 解析後的有向邊列表
    """

    nodes = paths.split("->")
    print("nodes:", nodes)
    edges = [(nodes[i], nodes[i+2]) for i in range(0, len(nodes) - 1, 2)]

    return edges

# 存放節點的固定座標

def save_positions(pos, position_file):
    """ 保存節點座標到 JSON 檔案 """
    with open(position_file, "w") as f:
        json.dump({k: v.tolist() for k, v in pos.items()}, f)

def load_positions(position_file):
    """ 從 JSON 檔案讀取節點座標 """
    try:
        with open(position_file, "r") as f:
            return {k: v for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}


def draw_subgraph(round_count, G_subgraph, paths, match_kg, path_num, process_name, position_file):
    # 創建資料夾
    folder = f"{date.today()}_{process_name}/"
    combined_folder_path = os.path.join(PATH_FOR_PYVIS, folder)
    if not os.path.exists(combined_folder_path):
        os.makedirs(combined_folder_path, exist_ok=True)

    highlight_path = parse_paths(paths) if paths else []
    net = Network(notebook=True, directed=True, height="750px", width="100%")
    
    # 重新計算座標，但保留之前的節點位置
    pos = nx.spring_layout(G_subgraph, seed=42)  # 這次生成的新佈局

    if position_file:
        position_file = os.path.join(PATH_FOR_PYVIS, position_file)
        # 嘗試載入之前的座標
        prev_positions = load_positions(position_file)
        for node in prev_positions:
            if node in pos:
                pos[node] = prev_positions[node]  # 保留舊座標
    else:
        position_file = f"{date.today()}_{process_name}/Q{round_count}.json"
        combined_pos_path = os.path.join(PATH_FOR_PYVIS, position_file)
        # 保存最新的節點座標（確保未來相同）
        save_positions(pos, combined_pos_path)

    for node in G_subgraph.nodes:
        if node not in pos:
            print(f"節點 {node} 沒有座標，跳過！")
            continue  # 確保節點有座標

        x, y = pos[node]  # 取出座標
        x, y = float(x) * 500, float(y) * 500  # 確保是浮點數並放大座標
        net.add_node(node, label=str(node), color='orange' if node in match_kg else 'lightblue',
                     size=20 if node in match_kg else 10, x=x, y=y, fixed=True)


    # 添加邊
    for edge in G_subgraph.edges:
        if edge in highlight_path:  # 高亮路徑
            net.add_edge(edge[0], edge[1], color="red", width=3, arrows="to")
        else:  # 普通邊
            net.add_edge(edge[0], edge[1], color="gray", width=1, arrows="to")
    
    # 關閉物理模擬
    net.toggle_physics(False)

    # file_paths = f"../pyvis_result/{date.today()}_{process_name}/"
    # if not os.path.exists(file_paths):
    #     os.mknod(file_paths)
    
    # 輸出為 HTML
    if path_num:
        net.show(f"/home/yuu0223/KG-Prompting-Based-on-Community-Search/pyvis_result/{date.today()}_{process_name}/Q{round_count}_{path_num}.html")
    else:
        net.show(f"/home/yuu0223/KG-Prompting-Based-on-Community-Search/pyvis_result/{date.today()}_{process_name}/Q{round_count}.html")


### ---
def draw_subgraph_only_paths(round_count, paths, match_kg, path_num, process_name, position_file):
    # 解析成三元組：node1, relation, node2
    elements = paths.split("->")
    triples = [(elements[i], elements[i+1], elements[i+2]) for i in range(0, len(elements) - 2, 2)]

    # 建立有向圖並加入邊
    G_subgraph = nx.DiGraph()
    for source, rel, target in triples:
        G_subgraph.add_edge(source, target, label=rel)

    # 創建資料夾
    folder = f"{date.today()}_{process_name}/"
    combined_folder_path = os.path.join(PATH_FOR_PYVIS, folder)
    if not os.path.exists(combined_folder_path):
        os.makedirs(combined_folder_path, exist_ok=True)

    highlight_path = parse_paths(paths) if paths else []
    net = Network(notebook=True, directed=True, height="750px", width="100%")
    
    # 重新計算座標，但保留之前的節點位置
    pos = nx.spring_layout(G_subgraph, seed=78)  # 這次生成的新佈局

    if position_file:
        position_file = os.path.join(PATH_FOR_PYVIS, position_file)
        # 嘗試載入之前的座標
        prev_positions = load_positions(position_file)
        for node in prev_positions:
            if node in pos:
                pos[node] = prev_positions[node]  # 保留舊座標
    else:
        position_file = f"{date.today()}_{process_name}/Q{round_count}.json"
        combined_pos_path = os.path.join(PATH_FOR_PYVIS, position_file)
        # 保存最新的節點座標（確保未來相同）
        save_positions(pos, combined_pos_path)

    for node in G_subgraph.nodes:
        if node not in pos:
            print(f"節點 {node} 沒有座標，跳過！")
            continue  # 確保節點有座標

        x, y = pos[node]  # 取出座標
        x, y = float(x) * 500, float(y) * 500  # 確保是浮點數並放大座標
        net.add_node(node, label=str(node), color='orange' if node in match_kg else 'lightblue',
                     size=20 if node in match_kg else 10, x=x, y=y, fixed=True, font={"size": 20})


    # 添加邊
    for edge in G_subgraph.edges:
        if edge in highlight_path:  # 高亮路徑
            net.add_edge(edge[0], edge[1], color="gray", width=3, arrows="to")
        else:  # 普通邊
            net.add_edge(edge[0], edge[1], color="gray", width=1, arrows="to")
    
    # 關閉物理模擬
    net.toggle_physics(False)

    # file_paths = f"../pyvis_result/{date.today()}_{process_name}/"
    # if not os.path.exists(file_paths):
    #     os.mknod(file_paths)
    
    # 輸出為 HTML
    if path_num:
        net.show(f"/home/yuu0223/KG-Prompting-Based-on-Community-Search/pyvis_result/{date.today()}_{process_name}/Q{round_count}_{path_num}.html")
    else:
        net.show(f"/home/yuu0223/KG-Prompting-Based-on-Community-Search/pyvis_result/{date.today()}_{process_name}/Q{round_count}.html")
### ---




def draw_subgraph_one_node(round_count, G_subgraph, paths, match_kg, process_name, position_file):
    # 創建資料夾
    folder = f"{date.today()}_{process_name}/"
    combined_folder_path = os.path.join(PATH_FOR_PYVIS, folder)
    if not os.path.exists(combined_folder_path):
        os.makedirs(combined_folder_path, exist_ok=True)

    highlight_path_list = []
    for i in range(len(paths)):
        highlight_path = parse_paths(paths[i]) if paths[i] else []
        highlight_path_list.append(highlight_path[0])

    net = Network(notebook=True, directed=True, height="750px", width="100%")
    
    # 重新計算座標，但保留之前的節點位置
    pos = nx.spring_layout(G_subgraph, seed=42)  # 這次生成的新佈局

    # if position_file:
    #     position_file = os.path.join(PATH_FOR_PYVIS, position_file)
    #     # 嘗試載入之前的座標
    #     prev_positions = load_positions(position_file)
    #     for node in prev_positions:
    #         if node in pos:
    #             pos[node] = prev_positions[node]  # 保留舊座標
    # else:
    #     position_file = f"{date.today()}_{process_name}/Q{round_count}.json"
    #     combined_pos_path = os.path.join(PATH_FOR_PYVIS, position_file)
    #     # 保存最新的節點座標（確保未來相同）
    #     save_positions(pos, combined_pos_path)

    for node in G_subgraph.nodes:
        if node not in pos:
            print(f"節點 {node} 沒有座標，跳過！")
            continue  # 確保節點有座標

        x, y = pos[node]  # 取出座標
        x, y = float(x) * 500, float(y) * 500  # 確保是浮點數並放大座標
        net.add_node(node, label=str(node), color='orange' if node in match_kg else 'lightblue',
                     size=20 if node in match_kg else 10, x=x, y=y, fixed=True)


    # 添加邊
    for edge in G_subgraph.edges:
        if edge in highlight_path_list:  # 高亮路徑
            net.add_edge(edge[0], edge[1], color="red", width=3, arrows="to")
        else:  # 普通邊
            net.add_edge(edge[0], edge[1], color="gray", width=1, arrows="to")
    
    # 關閉物理模擬
    net.toggle_physics(False)

    # file_paths = f"../pyvis_result/{date.today()}_{process_name}/"
    # if not os.path.exists(file_paths):
    #     os.mknod(file_paths)
    
    # 輸出為 HTML
    net.show(f"/home/yuu0223/KG-Prompting-Based-on-Community-Search/pyvis_result/{date.today()}_{process_name}/Q{round_count}.html")


# round_count = 374
# result_subgraph = {"Rectal_bleeding": ["Ischemia_of_the_bowel"],
#                    "Ischemia_of_the_bowel": ["Rectal_bleeding", "Constipation"],
#                    "Constipation": ["Ischemia_of_the_bowel", "Volvulus"],
#                    "Volvulus": ["Constipation", "Back_pain"],
#                    "Back_pain": ["Volvulus", "Spondylitis"],
#                    "Spondylitis": ["Back_pain", "Muscle_cramps"],
#                    "Muscle_cramps": ["Spondylitis"]}
# G_subgraph = nx.DiGraph()
# for node, neighbors in result_subgraph.items():  ### 可以替換成 graph_dict
#     for neighbor in neighbors:
#         G_subgraph.add_edge(node, neighbor)

# paths = "Rectal_bleeding->possible_disease->Ischemia_of_the_bowel->has_symptom->Constipation->possible_disease->Volvulus->has_symptom->Back_pain->possible_disease->Spondylitis->has_symptom->Muscle_cramps->possible_disease->Spondylitis->has_symptom->Back_pain->possible_disease->Volvulus->has_symptom->Constipation"
# match_kg = ['Rectal_bleeding', 'Muscle_cramps', 'Constipation']
# path_num=0
# process_name="GreedyDist+PR"
# position_file=None

# draw_subgraph(round_count, G_subgraph, paths, match_kg, path_num, process_name, position_file)




# def draw_subgraph_only_paths(round_count, paths, match_kg, path_num, process_name, position_file): 
#     # 解析成三元組：node1, relation, node2
#     elements = paths.split("->")
#     triples = [(elements[i], elements[i+1], elements[i+2]) for i in range(0, len(elements) - 2, 2)]

#     # 建立有向圖並加入邊
#     G_subgraph = nx.DiGraph()
#     for source, rel, target in triples:
#         G_subgraph.add_edge(source, target, label=rel)

#     # 創建資料夾
#     folder = f"{date.today()}_{process_name}/"
#     combined_folder_path = os.path.join(PATH_FOR_PYVIS, folder)
#     if not os.path.exists(combined_folder_path):
#         os.makedirs(combined_folder_path, exist_ok=True)

#     highlight_path = parse_paths(paths) if paths else []
#     net = Network(notebook=True, directed=True, height="900px", width="100%")  # 🔴 調大畫布

#     # 重新計算座標，但保留之前的節點位置 - 調整節點間距
#     pos = nx.spring_layout(G_subgraph, seed=78, k=2, iterations=50)  # 🔴 修改: 增加佈局參數
    
#     if position_file:
#         position_file = os.path.join(PATH_FOR_PYVIS, position_file)
#         # 嘗試載入之前的座標
#         prev_positions = load_positions(position_file)
#         for node in prev_positions:
#             if node in pos:
#                 pos[node] = prev_positions[node]  # 保留舊座標
#     else:
#         position_file = f"{date.today()}_{process_name}/Q{round_count}.json"
#         combined_pos_path = os.path.join(PATH_FOR_PYVIS, position_file)
#         # 保存最新的節點座標（確保未來相同）
#         save_positions(pos, combined_pos_path)
    
#     # 🔴 新增: 將所有節點排列在水平線上（不只是查詢節點）
#     all_nodes = list(G_subgraph.nodes())
#     if all_nodes:
#         y_level = 0  # 水平線的y座標
#         total_width = len(all_nodes) * 2.5  # 🔴 增加節點間距，避免重疊
#         start_x = -total_width / 2  # 起始x座標
        
#         # 將查詢節點優先排在前面
#         query_nodes_list = [node for node in all_nodes if node in match_kg]
#         other_nodes_list = [node for node in all_nodes if node not in match_kg]
#         ordered_nodes = query_nodes_list + other_nodes_list
        
#         for i, node in enumerate(ordered_nodes):
#             if node in pos:
#                 # 將所有節點均勻分佈在水平線上
#                 x_position = start_x + (i * 1.5) + 0.75
#                 pos[node] = (x_position, y_level)

#     # 🔴 修改: 定義顏色系統，與合併圖保持一致
#     colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
#     path_color = colors[path_num % len(colors)] if path_num is not None else 'red'

#     for node in G_subgraph.nodes:
#         if node not in pos:
#             print(f"節點 {node} 沒有座標，跳過！")
#             continue  # 確保節點有座標
        
#         x, y = pos[node]  # 取出座標
#         x, y = float(x) * 1000, float(y) * 1000  # 🔴 大幅放大座標，讓圖更大
        
#         net.add_node(node, 
#                     label=str(node), 
#                     color='orange' if node in match_kg else 'lightblue', 
#                     size=40 if node in match_kg else 25,  # 🔴 加大節點大小
#                     x=x, y=y, 
#                     fixed=True, 
#                     font={"size": 30})  # 🔴 加大字體

#     # 添加邊 - 🔴 修改: 使用與合併圖一致的顏色系統
#     for edge in G_subgraph.edges:
#         edge_data = G_subgraph[edge[0]][edge[1]]
#         label = edge_data.get('label', '')
        
#         if edge in highlight_path:  # 高亮路徑
#             net.add_edge(edge[0], edge[1], 
#                         color=path_color,  # 🔴 修改: 使用對應的路徑顏色
#                         width=6,  # 🔴 加粗線條
#                         arrows="to",
#                         title=f"Path {path_num+1 if path_num is not None else 1}: {label}")
#         else:  # 普通邊
#             net.add_edge(edge[0], edge[1], 
#                         color="lightgray",  # 🔴 修改: 使用淺灰色，與合併圖一致
#                         width=3,  # 🔴 加粗普通線條
#                         arrows="to",
#                         title=label)

#     # 關閉物理模擬
#     net.toggle_physics(False)

#     # 🔴 新增: 添加圖例說明
#     legend_html = f"""
#     <div style="position: absolute; top: 10px; right: 10px; background: white; padding: 10px; border: 1px solid black; border-radius: 5px;">
#         <h4>路徑 {path_num+1 if path_num is not None else 1}</h4>
#         <p><span style="color: orange;">●</span> 查詢節點</p>
#         <p><span style="color: lightblue;">●</span> 一般節點</p>
#         <p><span style="color: lightgray;">—</span> 一般邊</p>
#         <p><span style="color: {path_color};">—</span> 當前路徑</p>
#     </div>
#     """

#     # 輸出為 HTML
#     if path_num is not None:
#         output_path = f"/home/yuu0223/KG-Prompting-Based-on-Community-Search/pyvis_result/{date.today()}_{process_name}/Q{round_count}_{path_num}.html"
#     else:
#         output_path = f"/home/yuu0223/KG-Prompting-Based-on-Community-Search/pyvis_result/{date.today()}_{process_name}/Q{round_count}.html"
    
#     net.show(output_path)
    
#     # 🔴 新增: 在HTML中添加圖例
#     with open(output_path, 'r', encoding='utf-8') as f:
#         html_content = f.read()
    
#     # 在body標籤後插入圖例
#     html_content = html_content.replace('<body>', f'<body>{legend_html}')
    
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(html_content)