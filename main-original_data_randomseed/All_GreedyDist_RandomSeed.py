import numpy as np
import re
from neo4j import GraphDatabase, basic_auth
import pandas as pd
import pickle
import json
import os
import csv
import time
import random

import BuildDatabase
from llm import openrouter, gemini
from mindmap import Preprocessing
from communitysearch import FindKG, Greedy, GreedyDist, KGtoPath_PR, KGtoPath_BFS, PromptGenerate
import networkx as nx
from other import KG_vision_pyvis

from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="When cdn_resources is 'local'")

from tqdm import tqdm
import codecs

from time import perf_counter


if __name__ == "__main__":

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
    # session = driver.session()
    ### --- End of Step 1
    

    print("Start to match the entites...")
    ### --- 2. OpenAI API based keyword extraction and match entities
    GEMINI_API_KEY = os.getenv("gemini_api_key_yi")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of Step 2


    with open('./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyGen/E6_Hop3/20250804_BFS.csv', 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Q_ID', 'Question', 'Answer'])

    # with open('./output/node_amount.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(['Question_ID', 'Question', 'Greedy','GreedyDist', 'GreedyGen'])

    with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open('./data/chatdoctor5k/keyword_embeddings_new.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)


    docs_dir = './data/chatdoctor5k/document'
    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)
    
    ### --- 3. Extract Question Entities
    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    # with open("./output/MindMap_Paths_Number.json", "r", encoding='utf-8') as f1: ###
    #     data = json.load(f1)


    csv_rows = []
    json_data = []
    with open("./data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        lines = f.readlines()
        ### 設置隨機種子
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        # 總行數
        total_lines = len(lines)
        # 使用隨機種子選取 100 個題號
        selected_indices = random.sample(range(total_lines), 718)
        exclude = {552, 2553, 4333, 5084}  # 這裡是題號（從 1 開始）
        selected_indices = [q for q in selected_indices if q not in exclude]
        # selected_indices = selected_indices[249:250]
        print(f"隨機選取的題號: {selected_indices}")

        # for i in range(45, len(selected_indices)+1): 
        #     index = selected_indices[i]
        for count, index in enumerate(tqdm(selected_indices, desc="Processing Questions", unit="question", dynamic_ncols=True)):
            try:  
                line = lines[index]
                x = json.loads(line)
                input = x["qustion_output"]
                input = input.replace("\n","")
                input = input.replace("<OOS>","<EOS>")
                input = input.replace(":","") + "<END>"
                input_text = re.findall(re3,input)
                
                if input_text == []:
                    continue
                
                print('Question:', index+1,'\n',input_text[0])

                output = x["answer_output"]
                output = output.replace("\n","")
                output = output.replace("<OOS>","<EOS>")
                output = output.replace(":","") + "<END>"
                output_text = re.findall(re3,output)

                    
                question_kg = re.findall(re1,input)

                if len(question_kg) == 0:
                    question_kg = re.findall(re2,input)
                    if len(question_kg) == 0:
                        print("<Warning> no entities found", input)
                        # with open('./output/node_amount.csv', 'a+', newline='') as f6:
                        #     writer = csv.writer(f6)
                        #     writer.writerow([round_count, input_text[0],0,0,0])
                        #     f6.flush()
                        continue
                question_kg = question_kg[0].replace("<END>","").replace("<EOS>","")
                question_kg = question_kg.replace("\n","")
                question_kg = question_kg.split(", ")

                
                match_kg = []
                entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
            

                for kg_entity in question_kg:
                    
                    keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                    kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                    cos_similarities = Preprocessing.cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                    max_index = cos_similarities.argmax()
                            
                    match_kg_i = entity_embeddings["entities"][max_index]
                    while match_kg_i.replace(" ","_") in match_kg:
                        cos_similarities[max_index] = 0
                        max_index = cos_similarities.argmax()
                        match_kg_i = entity_embeddings["entities"][max_index]

                    match_kg.append(match_kg_i.replace(" ","_"))

                print('Question Entities:\n', match_kg, "\n") ### 與問題keywords相符的entities
                ### --- End of Step 3
                

                ### --- 4. Community Search
                start = perf_counter()
                graph_dict = FindKG.find_whole_KG(driver)
                # ---  4. Method 1: Community Search without size restriction
                # result_subgraph = Greedy.greedy_max_min_degree(graph_dict, match_kg)
                # # print("result_subgraph 1: \n", len(result_subgraph))
                # # greedy_node = len(result_subgraph) #計算節點用
                # ---

                # --- 4. Method 2: Community Search with size restriction!!
                # condition_constraint = {'distance':10, 'size':200}
                # distance, result_subgraph = GreedyDist.greedy_dist(graph_dict, match_kg, condition_constraint)
                # # print("result_subgraph 2: \n", len(result_subgraph))
                # # greedydist_node = len(result_subgraph)
                # ---
                

                # --- 4. Method 3: Community Search with size restriction (GreedyDist without check in the same subgraph 可斷開)
                # condition_constraint = {'distance':5, 'size':200}
                # distance_sub, result_subgraph_sub = GreedyDist.greedydist_subgraph(graph_dict, match_kg, condition_constraint)
                # print("result_subgraph 3: \n", len(result_subgraph_sub))
                # # greedydist_sub_node = len(result_subgraph_sub)
                # ---

                # --- 4. Method 4: Community Search without size restriction (GreedyGen)!!!
                condition_constraint = {'distance':3}
                result_subgraph = GreedyDist.greedy_gen(graph_dict, match_kg, condition_constraint)
                # print("result_subgraph 1: \n", len(result_subgraph))
                # greedydist_node = len(result_subgraph)
                # ---

                # --- 4. Method 5: Community Search without size restriction (GreedyGen 可斷開)
                # condition_constraint = {'distance':5}
                # result_subgraph_sub = GreedyDist.greedy_gen_subgraph(graph_dict, match_kg, condition_constraint)
                # print("result_subgraph 2: \n", len(result_subgraph_sub))
                # greedydist_sub_node = len(result_subgraph_sub)
                # ---
                step1_start = perf_counter()
                print(f"Find Subgraph: { step1_start - start:.3f} 秒")

                ### --- End of Step 4


                ### --- 5. Combine Edges to Paths
                # --- [需要時再使用] Key Entities sorted by PageRank value
                # # Calculate PageRank value of every nodes in Community Search subgraph (可以替換成整個KG - graph_dict)
                # G_graph = nx.DiGraph()
                # for node, neighbors in graph_dict.items():
                #     for neighbor in neighbors:
                #         G_graph.add_edge(node, neighbor)
                
                # pagerank_values_whole = nx.pagerank(G_graph, alpha=0.85)
                # # # 確認每個節點的 PageRank 值
                # # for node, rank in pagerank_values.items():
                # #     print(f"{node}: {rank}")

                # sorted_match_kg = sorted(match_kg, key=lambda x: pagerank_values_whole[x], reverse=True)
                # print("sorted_match :\n", sorted_match_kg)
                # match_kg = sorted_match_kg
                # # --- 
                # step2_start = perf_counter()
                # print(f"Sort match_kg: {step2_start - step1_start:.3f} 秒")
                ### ---

                # --- [需要時再使用] Key Entities sorted by Node Degree
                # G_graph = nx.DiGraph()
                # for node, neighbors in graph_dict.items():
                #     for neighbor in neighbors:
                #         G_graph.add_edge(node, neighbor)
                
                # degree_dict = dict(G_graph.degree())

                # sorted_match_kg = sorted(match_kg, key=lambda x: degree_dict[x], reverse=True)
                # print("sorted_match :\n", sorted_match_kg)
                # match_kg = sorted_match_kg
                # --- 

                # --- 5. Method 1: Double Way Paths 為每個關係建立雙向edges
                # community_result_path = FindKG.double_way_edge_finding(result_subgraph, driver)
                # print('community_result_path: \n', community_result_path)
                # path_join = PromptGenerate.JoinPath(path_list)
                # ---

                # --- 5. Method 2: Single Way Paths 為每個關係建立單向edges
                # path_list = FindKG.single_way_edge_finding(result_subgraph, driver)
                # print('single_path: \n',path_list, '\n')
                # print(len(path_list))
                # path_join = PromptGenerate.JoinPath(path_list)
                # ---

                # --- 5. Method 3: DFS & BFS (目前使用這個!)
                all_paths = FindKG.subgraph_path_finding(result_subgraph, match_kg)

                # result = [item for item in data if item.get("Q_ID") == index+1]
                # print("Q_ID from paths json:", result[0]["Q_ID"])

                # top_n = result[0]["mindmap"]
                top_n=10
                path_list, flag = KGtoPath_BFS.paths_in_neo4j_optimized_bfs_full(all_paths, top_n, driver)

                # # # 繪圖用
                # # G_subgraph = nx.DiGraph()
                # # for node, neighbors in result_subgraph.items():  
                # #     for neighbor in neighbors:
                # #         G_subgraph.add_edge(node, neighbor)
                # ---

                # --- 5. Method 4: Average PageRank
                # # Calculate PageRank value of every nodes in Community Search subgraph (可以替換成整個KG - graph_dict)
                # G_subgraph = nx.DiGraph()
                # for node, neighbors in result_subgraph.items():  ### 可以替換成 graph_dict
                #     for neighbor in neighbors:
                #         G_subgraph.add_edge(node, neighbor)
                
                # pagerank_values = nx.pagerank(G_subgraph, alpha=0.85)

                # # result = [item for item in data if item.get("Q_ID") == index+1]
                # # print("Q_ID from paths json:", result[0]["Q_ID"])

                # # top_n = result[0]["mindmap"]
                # top_n=10
                # path_list, flag = KGtoPath_PR.optimized_community_search_workflow(
                #     result_subgraph, #計算PR用的
                #     result_subgraph, 
                #     match_kg, 
                #     driver, 
                #     top_n=top_n)
                # ---
                ### --- End of Step 5
                step3_start = perf_counter()
                print(f"Cal PageRank: {step3_start - step1_start:.3f} 秒")



                ### --- 6. Choose Paths
                # --- 6. Method 1: 將每回合的每條 paths join by "->" (依舊維持多條，沒有合併成一大條)
                # path_join = [] 
                # for cur_list in path_list:
                #     join_list = "->".join(cur_list)
                #     path_join.append(join_list)
                # path_join = "\n\n\n".join(path_join)
                # ---

                # --- 6. Method 2: 將每回合的每條 paths 合併成一大條 path
                # if len(path_list) > 1:
                #     path_join = []
                #     for list in path_list:
                #         join_list = "->".join(list)
                #         path_join.append(join_list)
                #     path_join = "\n\n\n".join(path_join)
                # elif len(path_list) == 1:
                #     path_join = "->".join(path_list[0])
                # else:
                #     path_join = []
                # print("path: ", path_join)
                # ---

                # --- 6. Method 3: 排列組合 & 選取字數最短的 top N 條 (目前使用這個!)
                (path_join, 
                 path_join_list, 
                 path_nodes_count) = FindKG.combine_lists(community_search_paths=path_list, pagerank_values=None, top_n=top_n, flag=flag)
                # For PageRank
                # (path_join, 
                # path_join_list, 
                # path_nodes_count) = FindKG.combine_lists(path_list, pagerank_values, top_n, flag)
                # # print("path_nodes_count: ", path_nodes_count)
                # ---
                ### --- End of Step 6
                step4_start = perf_counter()
                print(f"Find Paths: {step4_start - step3_start:.3f} 秒")


                ### --- 7. 將 Paths 轉換為 Natural Language
                prompt = PromptGenerate.GeneratePathPrompt(path_join, chat_gm)
                step5_start = perf_counter()
                print(f"Generate Paths in NL: {step5_start - step4_start:.3f} 秒")
                ### --- End of Step 7


                ### --- 8. Visualization Subgraph of Community Search
                # process_name = "GreedyGen+PR"
                # # 目前作法是一條 path 一張圖 (之後想改成多條 paths 一張圖!)
                # if flag == "one_node":
                #     # position_file = f"{date.today()}_{process_name}/Q{round_count}.json"
                #     KG_vision_pyvis.draw_subgraph_one_node(index+1, G_subgraph, path_join_list, match_kg, process_name, position_file=None)
                # else:
                #     for i in range(len(path_join_list)):
                #         KG_vision_pyvis.draw_subgraph(index+1, G_subgraph, path_join_list[i], match_kg, i, process_name, position_file=None)
                ### --- End of Step 8
                
                ### --- 9. KG-based Prompt Generation
                # 若沒有回答出來，讓 LLM 重新回答
                times=1
                while True:
                    output_all = PromptGenerate.final_answer(input_text[0], prompt, chat_gm)
                    output1 = PromptGenerate.extract_final_answer(output_all)
                    # output1, output2, output3 = extract_final_answer(output_all)
                    if len(output1) > 0:
                        output1 = output1[0]
                        # output2 = output2[0]
                        # output3 = output3[0]
                        break
                    elif times == 2:
                        break
                    else:
                        times+=1
                print("Final Answer: ", output1)
                # if is_unable_to_answer(output_all, chat_gm):
                #     output_all = PromptGenerate.final_answer(input_text[0],prompt, chat_gm)
                # print(output1)
                # time.sleep(60)
                step6_start = perf_counter()
                print(f"Generate Final Answer: {step6_start - step5_start:.3f} 秒")
                ### --- End of Step 9 


                ### --- 10. 暫存結果
                # csv_rows.append([index+1, input_text[0], output1])
                with open('./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyGen/E6_Hop3/20250804_BFS.csv', 'a+', newline='') as f4:
                    writer = csv.writer(f4)
                    writer.writerow([index+1,  input_text[0], output1])
                    f4.flush()
                # ### --- save for json
                # json_data.append({
                #     "Q_ID": index+1,
                #     "question": input_text[0],
                #     "query_nodes": match_kg,
                #     "subgraph": result_subgraph,
                #     "paths-list": path_join_list,
                #     "output": output1
                # })
                # ### --- End of Step 10
            
            except Exception as e:
                print(f"錯誤在第 {index+1} 題: {e}")

    # 8. 統一寫入檔案
    # with open('./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/E6_Hop10/20250729_BFS.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerows(csv_rows)

    # with open("./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/E6_Hop10/20250729_BFS.json", "w", encoding="utf-8") as f:
    #     json.dump(json_data, f, ensure_ascii=False, indent=4)

    driver.close()






