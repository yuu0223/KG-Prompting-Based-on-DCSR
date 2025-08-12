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



def draw_subgraph_combined_paths(round_count, path_list, match_kg, process_name, position_file=None):
    """
    合併多條路径到一張圖中顯示
    
    Args:
        round_count: 輪次計數
        path_list: 路径列表 (多條路径)
        match_kg: 匹配的節點
        process_name: 處理名稱
        position_file: 位置檔案路径
    """
    
    # 建立有向圖
    G_subgraph = nx.DiGraph()
    all_highlight_paths = []
    
    # 處理所有路径
    for path_idx, paths in enumerate(path_list):
        if not paths:
            continue
            
        # 解析成三元組：node1, relation, node2
        elements = paths.split("->")
        triples = [(elements[i], elements[i+1], elements[i+2]) for i in range(0, len(elements) - 2, 2)]
        
        # 加入邊到圖中
        for source, rel, target in triples:
            # 如果邊已存在，可以選擇保留或更新關係標籤
            if G_subgraph.has_edge(source, target):
                # 如果已有邊，可以合併標籤或保留原有
                existing_label = G_subgraph[source][target].get('label', '')
                if rel not in existing_label:
                    new_label = f"{existing_label}, {rel}" if existing_label else rel
                    G_subgraph[source][target]['label'] = new_label
            else:
                G_subgraph.add_edge(source, target, label=rel)
        
        # 解析路径用於高亮顯示
        highlight_path = parse_paths(paths) if paths else []
        all_highlight_paths.extend(highlight_path)
    
    # 創建資料夾
    folder = f"{date.today()}_{process_name}/"
    combined_folder_path = os.path.join(PATH_FOR_PYVIS, folder)
    if not os.path.exists(combined_folder_path):
        os.makedirs(combined_folder_path, exist_ok=True)
    
    # 創建網路圖
    net = Network(notebook=True, directed=True, height="750px", width="100%")
    
    # 計算或載入節點座標 - 調整節點間距
    pos = nx.spring_layout(G_subgraph, seed=78, k=2, iterations=50)
    
    if position_file:
        position_file_path = os.path.join(PATH_FOR_PYVIS, position_file)
        prev_positions = load_positions(position_file_path)
        for node in prev_positions:
            if node in pos:
                pos[node] = prev_positions[node]
    else:
        position_file = f"{date.today()}_{process_name}/Q{round_count}_combined.json"
        combined_pos_path = os.path.join(PATH_FOR_PYVIS, position_file)
        save_positions(pos, combined_pos_path)
    
    # ### --- 將查詢節點排列在同一條水平線上
    # query_nodes_list = list(match_kg)
    # if query_nodes_list:
    #     y_level = 0  # 水平線的y座標
    #     total_width = len(query_nodes_list) * 1.5  # 縮短查詢節點間距
    #     start_x = -total_width / 2  # 起始x座標
        
    #     for i, node in enumerate(query_nodes_list):
    #         if node in pos:
    #             # 將查詢節點均勻分佈在水平線上，間距縮短
    #             x_position = start_x + (i * 1.5) + 0.75
    #             pos[node] = (x_position, y_level)
    
    # 添加節點
    for node in G_subgraph.nodes:
        if node not in pos:
            print(f"節點 {node} 沒有座標，跳過！")
            continue
        
        x, y = pos[node]
        x, y = float(x) * 500, float(y) * 500
        
        # 根據節點是否在查詢中設置顏色和大小
        node_color = 'orange' if node in match_kg else 'lightblue'
        node_size = 20 if node in match_kg else 10
        
        net.add_node(node, 
                    label=str(node), 
                    color=node_color, 
                    size=node_size, 
                    x=x, y=y, 
                    fixed=True, 
                    font={"size": 20})
    
    # 添加邊，用不同顏色區分不同路径
    colors = ['green', '#C2EABD', '#C0BABC', '#C7AC92', '#CD533B', '#37392E', '#19647E', '#28AFB0', 'gold', '#745C97']
    
    # 先添加所有普通邊（灰色）
    for edge in G_subgraph.edges:
        edge_data = G_subgraph[edge[0]][edge[1]]
        label = edge_data.get('label', '')
        net.add_edge(edge[0], edge[1], 
                    color="lightgray", 
                    width=1, 
                    arrows="to",
                    title=label)  # 鼠標懸停顯示關係
    
    # 然後添加高亮路径邊，用不同顏色標示不同路径
    for path_idx, paths in enumerate(path_list):
        if not paths:
            continue
            
        highlight_path = parse_paths(paths) if paths else []
        path_color = colors[path_idx % len(colors)]  # 循環使用顏色
        
        for edge in highlight_path:
            if G_subgraph.has_edge(edge[0], edge[1]):
                edge_data = G_subgraph[edge[0]][edge[1]]
                label = edge_data.get('label', '')
                net.add_edge(edge[0], edge[1], 
                            color=path_color, 
                            width=3, 
                            arrows="to",
                            title=f"Path {path_idx+1}: {label}")
    
    # 關閉物理模擬
    net.toggle_physics(False)
    
    # 添加圖例說明
    legend_html = """
    <div style="position: absolute; top: 10px; right: 10px; background: white; padding: 10px; border: 1px solid black; border-radius: 5px;">
        <h4>圖例</h4>
        <p><span style="color: orange;">●</span> 查詢節點</p>
        <p><span style="color: lightblue;">●</span> 一般節點</p>
        <p><span style="color: lightgray;">—</span> 一般邊</p>
    """
    
    for i in range(min(len(path_list), len(colors))):
        if i < len(path_list) and path_list[i]:  # 確保路径存在
            legend_html += f'<p><span style="color: {colors[i]};">—</span> 路径 {i+1}</p>'
    
    legend_html += "</div>"
    
    # 輸出為 HTML
    output_path = f"/home/yuu0223/KG-Prompting-Based-on-Community-Search/pyvis_result/{date.today()}_{process_name}/Q{round_count}_combined_paths.html"
    net.show(output_path)
    
    # 可選：在HTML中添加圖例
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 在body標籤後插入圖例
    html_content = html_content.replace('<body>', f'<body>{legend_html}')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"已生成合併路径圖: {output_path}")
    return output_path