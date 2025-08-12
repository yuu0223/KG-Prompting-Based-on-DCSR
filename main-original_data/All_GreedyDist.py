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
from datetime import date

import BuildDatabase
from llm import openrouter, gemini
from mindmap import Preprocessing
from communitysearch import FindKG, Greedy, GreedyDist, PromptGenerate
import networkx as nx
from other import KG_vision_pyvis

from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")



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
    GEMINI_API_KEY = os.getenv("gemini_api_key")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of Step 2


    # with open('./output/CommunitySearch/CS_Greedy_BFS_100.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(['Question_ID', 'Question', 'Reference_Ans', 'Answer'])

    # with open('./output/paths_node_amount_GreedyGenwithBFS.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(['Question_ID', 'Question', 'GreedyGen+BFS'])
 
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
    with open("./data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        round_count = 3872 ## Initial : 30
        for line in f.readlines()[round_count:round_count+1]:
            round_count+=1
            x = json.loads(line)
            input = x["qustion_output"]
            input = input.replace("\n","")
            input = input.replace("<OOS>","<EOS>")
            input = input.replace(":","") + "<END>"
            input_text = re.findall(re3,input)
            
            if input_text == []:
                continue
            
            print('Question:', round_count,'\n',input_text[0])

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

            answer_kg = re.findall(re1,output)
            if len(answer_kg) == 0:
                answer_kg = re.findall(re2,output)
                if len(answer_kg) == 0:
                    print("<Warning> no entities found", output)
                    # with open('./output/node_amount.csv', 'a+', newline='') as f6:
                    #     writer = csv.writer(f6)
                    #     writer.writerow([round_count, input_text[0],-1,-1,-1])
                    #     f6.flush()
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
            

            ### --- 4. Community Search
            graph_dict = FindKG.find_whole_KG(driver)
            # ---  4. Method 1: Community Search without size restriction
            # result_subgraph = Greedy.greedy_max_min_degree(graph_dict, match_kg)
            # print("result_subgraph 1: \n", len(result_subgraph))
            # # greedy_node = len(result_subgraph) #計算節點用
            # ---

            # --- 4. Method 2: Community Search with size restriction
            condition_constraint = {'distance':5, 'size':200}
            distance, result_subgraph = GreedyDist.greedy_dist(graph_dict, match_kg, condition_constraint)
            # print("result_subgraph 2: \n", len(result_subgraph))
            # greedydist_node = len(result_subgraph)
            # ---

            # --- 4. Method 3: Community Search with size restriction (GreedyDist without check in the same subgraph 可斷開)
            # condition_constraint = {'distance':5, 'size':200}
            # distance_sub, result_subgraph_sub = GreedyDist.greedydist_subgraph(graph_dict, match_kg, condition_constraint)
            # print("result_subgraph 3: \n", len(result_subgraph_sub))
            # greedydist_sub_node = len(result_subgraph_sub)
            # ---

            # --- 4. Method 4: Community Search without size restriction (GreedyGen)
            # condition_constraint = {'distance':5}
            # result_subgraph = GreedyDist.greedy_gen(graph_dict, match_kg, condition_constraint)
            # # print("result_subgraph 1: \n", len(result_subgraph))
            # # greedygen_node = len(result_subgraph)
            # ---

            # --- 4. Method 5: Community Search without size restriction (GreedyGen 可斷開)
            # condition_constraint = {'distance':5}
            # result_subgraph_sub = GreedyDist.greedy_gen_subgraph(graph_dict, match_kg, condition_constraint)
            # print("result_subgraph 2: \n", len(result_subgraph_sub))
            # greedygen_sub_node = len(result_subgraph_sub)
            # ---
            ### --- End of Step 4


            ### --- 5. Combine Edges to Paths
            # --- [需要時再使用] Key Entities sorted by PageRank value
            # Calculate PageRank value of every nodes in Community Search subgraph (可以替換成整個KG - graph_dict)
            # G_subgraph = nx.DiGraph()
            # for node, neighbors in result_subgraph.items():
            #     for neighbor in neighbors:
            #         G_subgraph.add_edge(node, neighbor)
            
            # pagerank_values = nx.pagerank(G_subgraph, alpha=0.85)
            # 確認每個節點的 PageRank 值
            # for node, rank in pagerank_values.items():
            #     print(f"{node}: {rank}")

            # sorted_match_kg = sorted(match_kg, key=lambda x: pagerank_values[x], reverse=True)
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
            # all_paths = FindKG.subgraph_path_finding(result_subgraph, match_kg)
            # top_n = 10
            # path_list, flag  = FindKG.paths_in_neo4j(all_paths, top_n, driver)

            # # 繪圖用
            # G_subgraph = nx.DiGraph()
            # for node, neighbors in result_subgraph.items():  
            #     for neighbor in neighbors:
            #         G_subgraph.add_edge(node, neighbor)
            # ---

            # --- 5. Method 4: Average PageRank
            # Calculate PageRank value of every nodes in Community Search subgraph (可以替換成整個KG - graph_dict)
            G_subgraph = nx.DiGraph()
            for node, neighbors in result_subgraph.items():  ### 可以替換成 graph_dict
                for neighbor in neighbors:
                    G_subgraph.add_edge(node, neighbor)
            
            pagerank_values = nx.pagerank(G_subgraph, alpha=0.85)
            # 確認每個節點的 PageRank 值
            # for node, rank in pagerank_values.items():
            #     print(f"{node}: {rank}")

            all_paths = FindKG.subgraph_path_finding(result_subgraph, match_kg) ### symptoms_match_kg替換，依照PR值來排序
            top_n = 10
            path_list, flag = FindKG.paths_in_neo4j_for_PageRank(all_paths, pagerank_values, top_n, driver)
            # print("all_paths: \n", all_paths)
            # print("path_list: \n", path_list)
            # ---
            ### --- End of Step 5

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
            # (path_join, 
            #  path_join_list, 
            #  path_nodes_count) = FindKG.combine_lists(community_search_paths=path_list, pagerank_values=None, top_n=top_n, flag=flag)
            # For PageRank
            (path_join, 
             path_join_list, 
             path_nodes_count) = FindKG.combine_lists(path_list, pagerank_values, top_n, flag)
            # print("path_join: ", path_join_list)
            # print("path_nodes_count: ", path_nodes_count)

            # data = {round_count: {"path_join: ": path_join, "path_join_list": path_join_list}}
            # # 讀取現有 JSON
            # if os.path.exists("./output/CommunitySearch/GreedyGenwithBFS.json"):
            #     with open("./output/CommunitySearch/GreedyGenwithBFS.json", "r", encoding="utf-8") as f:
            #         try:
            #             data_dict = json.load(f)  # 讀取 JSON 字典
            #         except json.JSONDecodeError:
            #             data_dict = {}  # 如果檔案有問題，則重設為空字典
            # else:
            #     data_dict = {}

            # # 確保 JSON 是 `dict`，然後加入新資料
            # if isinstance(data_dict, dict):
            #     data_dict.update(data)  # 加入新的 key-value

            # # 寫回 JSON 檔案
            # with open("./output/CommunitySearch/GreedyGenwithBFS.json", "w", encoding="utf-8") as f:
            #     json.dump(data_dict, f, ensure_ascii=False, indent=4)
            # ---
            ### --- End of Step 6

            ### --- 7. 將 Paths 轉換為 Natural Language
            prompt = PromptGenerate.GeneratePathPrompt(path_join, chat_gm)
            ### --- End of Step 7


            ### --- 8. Visualization Subgraph of Community Search
            # ### GreedyDist、GreedyGen專用
            # process_name = "GreedyGen+BFS"
            # # 目前作法是一條 path 一張圖 (之後想改成多條 paths 一張圖!)
            # if flag == "one_node":
            #     # position_file = f"{date.today()}_{process_name}/Q{round_count}.json"
            #     KG_vision_pyvis.draw_subgraph_one_node(round_count, G_subgraph, path_join_list, match_kg, process_name, position_file=None)
            # else:
            #     for i in range(len(path_join_list)):
            #         KG_vision_pyvis.draw_subgraph(round_count, G_subgraph, path_join_list[i], match_kg, i, process_name, position_file=None)

            # ### GreedyDistSub、GreedyGenSub專用
            # process_name = "GreedyDistSub+PR"
            # position_process = "GreedyDist+PR"
            # # process_name = "GreedyGenSub+PR"
            # # position_process = "GreedyGen+PR"
            # # 目前作法是一條 path 一張圖 (之後想改成多條 paths 一張圖!)
            # position_file = f"{date.today()}_{position_process}/Q{round_count}.json"
            # for i in range(len(path_join_list)):
            #     KG_vision_pyvis.draw_subgraph(round_count, G_subgraph, path_join_list[i], match_kg, i, process_name, position_file)
        
            # ### 沒有path的專用
            # KG_vision_pyvis.draw_subgraph(round_count, G_subgraph, None, match_kg, None, process_name)
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
            # print(output1)
            # time.sleep(60)
            ### --- End of Step 9


            ### --- 10. Save the output
            ### --- save for json
            json_data.append({
                "Q_ID": round_count,
                "question": input_text[0],
                "query_nodes": match_kg,
                "subgraph_dict": result_subgraph,
                "paths-list": path_join_list,
                "output": output1
            })

            # with open("./output/CommunitySearch/Final_Q714/GreedyDist/PR_Last.json", "w", encoding="utf-8") as f:
            #     json.dump(json_data, f, ensure_ascii=False, indent=4)

            with open('./output/CommunitySearch/Final_Q714/GreedyDist/20250408_PR_Gemini20Flash.csv', 'a+', newline='') as f6:
                writer = csv.writer(f6)
                writer.writerow([round_count, input_text[0], output_text[0], output1])
                f6.flush()

            
            # with open('./output/paths_node_amount_GreedyGenwithBFS.csv', 'a+', newline='') as f6:
            #     writer = csv.writer(f6)
            #     writer.writerow([round_count, input_text[0], path_nodes_count])
            #     f6.flush()

            # time.sleep(5)
            ### --- End of Step 10
    
    with open("./output/CommunitySearch/Final_Q714/GreedyDist/PR_Last.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    driver.close()






