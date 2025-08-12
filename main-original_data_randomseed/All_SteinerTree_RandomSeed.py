import numpy as np
import re
from neo4j import GraphDatabase
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
from communitysearch import FindKG, PromptGenerate, SteinerTree, KGtoPath_PR, KGtoPath_BFS
from networkx.algorithms.approximation import steiner_tree
import networkx as nx
from other import KG_vision_pyvis

from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm



if __name__ == "__main__":

    ### ---- 1. build neo4j knowledge graph datasets
    uri = os.getenv("neo4j_uri")
    username = os.getenv("neo4j_username")
    password = os.getenv("neo4j_password")
    print(uri) # 檢查用

    # --- build KG 
    # data_path = './data/chatdoctor5k/train.txt'
    # BuildDatabase.BuildNeo4j(data_path)
    # ---

    driver = GraphDatabase.driver(uri, auth=(username, password))
    # session = driver.session()
    ### --- End of Step 1
    

    print("Start to match the entites...")
    ### --- 2. OpenAI API based keyword extraction and match entities
    GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade_2")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of Step 2


    # with open('./output/SteinerTree/Final_Q714_Gemini1.5/E3_Use_whole_KG/20250528_PR.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(['Q_ID', 'Question','Answer'])

    # with open('./output/SteinerTree/node_amount_BFS.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(['Question_ID', 'Question', 'KG_nodes_count', 'Path_nodes_count'])


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

    json_data = []
    SAVE_INTERVAL = 5
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
        selected_indices = selected_indices[251:252]
        print(f"隨機選取的題號: {selected_indices}")

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
                        continue
                question_kg = question_kg[0].replace("<END>","").replace("<EOS>","")
                question_kg = question_kg.replace("\n","")
                question_kg = question_kg.split(", ")

                answer_kg = re.findall(re1,output)
                if len(answer_kg) == 0:
                    answer_kg = re.findall(re2,output)
                    if len(answer_kg) == 0:
                        print("<Warning> no entities found", output)
                        continue
                answer_kg = answer_kg[0].replace("<END>","").replace("<EOS>","")
                answer_kg = answer_kg.replace("\n","")
                answer_kg = answer_kg.split(", ")

                
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
                

                ### --- 4. Steiner Tree
                graph_dict = FindKG.find_whole_KG(driver)
                #1123個node
                if len(match_kg) <= 1:
                    path_join = ["There's no any paths."] ##### ---- 這邊可以考慮找neighbors?
                else:
                    original_G = SteinerTree.create_graph(graph_dict)
                    approx_steiner_tree = steiner_tree(original_G, match_kg)

                    ### --- [需要時再用] 視覺化
                    # SteinerTree.steiner_tree_plt(approx_steiner_tree, approx_steiner_tree, index)
                    ### ---
                    
                    ### --- 4. Method 1: [shortest path] 依照 query nodes 順序來尋找路徑 (有overlap)
                    # all_nodes_list = SteinerTree.get_query_nodes_path_nodes(approx_steiner_tree, match_kg)
                    # # print("test: ",all_nodes_list)
                    # # 依照path nodes的順序尋找relations
                    # all_path_list = (SteinerTree.get_relations_with_nodes(all_nodes_list, driver) if len(all_nodes_list) > 0 else [])
                    # print("Original Steiner Tree paths:", all_path_list)
                    # # 用"->"把路徑都串起來
                    # path_join = (SteinerTree.join_all_path(all_path_list) if len(all_path_list) > 0 else [])
                    ### ---

                    ### --- 4. Method 2: [shortest path] 不管 key entities 順序，依照 shortest paths 來尋找路徑 (無overlap)
                    # all_nodes_list = SteinerTree.get_shortest_path_nodes(original_G, approx_steiner_tree, match_kg)
                    # # 依照path nodes的順序尋找relations
                    # all_path_list = (SteinerTree.get_relations_with_nodes(all_nodes_list, driver) if len(all_nodes_list) > 0 else [])
                    # # 用"->"把路徑都串起來
                    # path_join = (SteinerTree.join_all_path(all_path_list) if len(all_path_list) > 0 else [])
                    ### ---

                    ### --- 4. Method 3: [shortest path] Key Entities sorted by PageRank
                    # # PageRank
                    # pagerank_values = nx.pagerank(approx_steiner_tree, alpha=0.85)
                    # # 每個節點的 PageRank 值
                    # # for node, rank in pagerank_values.items():
                    # #     print(f"{node}: {rank}")
                    # sorted_match_kg = sorted(match_kg, key=lambda x: pagerank_values[x], reverse=True)
                    # print("sorted_match :\n", sorted_match_kg)

                    # all_nodes_list = SteinerTree.get_query_nodes_path_nodes(approx_steiner_tree, sorted_match_kg)
                    # # 依照path nodes的順序尋找relations
                    # all_path_list = (SteinerTree.get_relations_with_nodes(all_nodes_list, driver) if len(all_nodes_list) > 0 else [])
                    # # 用"->"把路徑都串起來
                    # path_join = (SteinerTree.join_all_path(all_path_list) if len(all_path_list) > 0 else [])
                    ### ---

                    ### --- 4. Method 4: [BFS] 依照 query nodes 順序來尋找路徑
                    # # 將graph轉換成dict
                    # result_subgraph = nx.to_dict_of_lists(approx_steiner_tree)
                    # # BFS開始尋找paths
                    # all_paths = FindKG.subgraph_path_finding(result_subgraph, match_kg)
                    # top_n = 10
                    # # path_list, flag = FindKG.paths_in_neo4j(all_paths, top_n, driver)
                    # path_list, flag = KGtoPath_BFS.paths_in_neo4j_optimized_bfs
                    # # 排列組合 & 選取字數最短的 top N 條
                    # path_join, path_join_list, path_nodes_count = FindKG.combine_lists(community_search_paths=path_list, pagerank_values=None, top_n=top_n, flag=flag)
                    # # print("Steiner Tree paths: ", path_join_list)
                    ### ---

                    ### --- 4. Method 5: [Average PageRank] 依照 query nodes 順序來尋找路徑
                    # # 將graph轉換成dict
                    result_subgraph = nx.to_dict_of_lists(approx_steiner_tree)
                    # print("result_subgraph: \n", len(result_subgraph))
                    # Calculate PageRank value of every nodes in Community Search subgraph (可以替換成整個KG - graph_dict) 
                    pagerank_values = nx.pagerank(original_G, alpha=0.85)
                    # 確認每個節點的 PageRank 值
                    # for node, rank in pagerank_values.items():
                    #     print(f"{node}: {rank}")

                    all_paths = FindKG.subgraph_path_finding(result_subgraph, match_kg) ### symptoms_match_kg替換，依照PR值來排序
                    top_n = 10
                    # path_list, flag = FindKG.paths_in_neo4j_for_PageRank(all_paths, pagerank_values, top_n, driver)
                    path_list, flag = KGtoPath_PR.optimized_community_search_workflow(
                    graph_dict,
                    result_subgraph, 
                    match_kg, 
                    driver, 
                    top_n=top_n)
                    # print("all_paths: \n", all_paths)
                    # print("path_list: \n", path_list)

                    # 排列組合 & 選取字數最短的 top N 條
                    path_join, path_join_list, path_nodes_count = FindKG.combine_lists(path_list, pagerank_values, top_n, flag)
                    # print("Steiner Tree paths: ", path_join_list)
                    print("Path nodes count: \n", path_nodes_count)
                    ### ---
                ### --- End of Step 4


                ### --- 7. 將 Paths 轉換為 Natural Language
                prompt = PromptGenerate.GeneratePathPrompt(path_join, chat_gm)
                # # print(prompt)
                ### --- End of Step 7


                ### --- 8. Visualization Subgraph of Steiner Tree
                # # 目前作法是一條 path 一張圖 (之後想改成多條 paths 一張圖!)
                # for i in range(len(path_join_list)):
                #     KG_vision_pyvis.draw_subgraph(round_count, G_subgraph, path_join_list[i], match_kg, i)
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

                # if is_unable_to_answer(output_all, chat_gm):
                #     output_all = PromptGenerate.final_answer(input_text[0],prompt, chat_gm)

                # time.sleep(60)
                ### --- End of Step 9


                ### --- 10. Save the output
                ### --- save for json
                json_data.append({
                    "Q_ID": index+1,
                    "question": input_text[0],
                    "query_nodes": match_kg,
                    "paths-list": path_join_list,
                    "output": output1
                })

                # # 每 SAVE_INTERVAL 筆就存檔一次
                # if (count + 1) % SAVE_INTERVAL == 0:
                #     with open("./output/SteinerTree/Final_Q714/Last_PR.json", "w", encoding="utf-8") as f:
                #         json.dump(json_data, f, ensure_ascii=False, indent=4)
                # ### ---

                # with open('./output/SteinerTree/Final_Q714_Gemini1.5/E3_Use_whole_KG/20250528_PR.csv', 'a+', newline='') as f6:
                #     writer = csv.writer(f6)
                #     writer.writerow([index+1, input_text[0], output1])
                #     f6.flush()

                # with open('./output/SteinerTree/Final_Q714_Gemini1.5/E3_Use_whole_KG/20250528_PR.csv', 'a+', newline='') as f6:
                #     writer = csv.writer(f6)
                #     writer.writerow([index+1, input_text[0], len(result_subgraph), path_nodes_count])
                #     f6.flush()

                # # time.sleep(5)
                ### --- End of Step 10


            except Exception as e:
                print(f"錯誤在第 {index+1} 題: {e}")

    with open("./output/run.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    driver.close()