### --- 這份檔案是給執行完CommunitySearch找出路徑後的Gemini產生回答階段
### 1086

import json
from communitysearch import PromptGenerate
import os
import csv
import re
from llm import gemini
import random
import ollama
from tqdm import tqdm
from time import sleep

### --- 2. OpenAI API based keyword extraction and match entities
GEMINI_API_KEY = os.getenv("gemini_api_key_yi")
chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
### --- End of Step 2

# with open('../output/CommunitySearch/Final_Q714_Gemma3-1B/GreedyDist/20250519_PR.csv', 'w', newline='') as f4: ###
#     writer = csv.writer(f4)
#     writer.writerow(['Q_ID','Question','Reference_Answer','Answer'])

# with open('./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/E2_Path_MM/20250527_BFS_new.csv', 'w', newline='') as f4:
#     writer = csv.writer(f4)
    # writer.writerow(['Question_ID', 'Question', 'Answer'])


re3 = r"<CLS>(.*?)<SEP>"

json_data = []
# SAVE_INTERVAL = 5
# with open("../data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
#     round_count = 0 ## Initial : 30
#     for line in f.readlines()[round_count:]:
#         round_count+=1
#         x = json.loads(line)
#         answer_output = x["answer_output"]
#         print("Q_ID from data:", x["Q_ID"])

with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

round_count = 0

# json_data = []
# SAVE_INTERVAL = 5
for line in tqdm(lines[round_count:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
    x = json.loads(line)
    answer_output = x["answer_output"]
    print("Q_ID from data:", x["Q_ID"])

    with open("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/20250505_BFS_Gemini15.json", "r", encoding='utf-8') as f1: ###
        data = json.load(f1)
        id = x["Q_ID"]
        result = [item for item in data if item.get("Q_ID") == id]
        print("Q_ID from paths json:", result[0]["Q_ID"])
        # print("result:", result)

        # print(result[0]["paths-list"]) if result else print(f"Q_ID={id} is not found.")
        
        if result :
            if result[0]["paths-list"] == []:
                path_join = "There's no any reference path in this case."
            else:
                paths = result[0]["paths-list"][0]
                print("paths:", paths)
                path_join = "\n\n\n".join(paths)
        else:
            print(f"Q_ID={id} is not found.")
            continue
        

        ### --- 7. 將 Paths 轉換為 Natural Language
        prompt = PromptGenerate.GeneratePathPrompt(path_input=path_join, chat_gm=chat_gm)
        print("prompt: \n", prompt)
        ### --- End of Step 7

        ### --- 9. KG-based Prompt Generation
        # 若沒有回答出來，讓 LLM 重新回答
        times=1
        while True:
            output_all = PromptGenerate.final_answer(question=result[0]["question"], prompt_path=prompt, chat_gm=chat_gm)
            print("output_all: \n", output_all)
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
                sleep(3)
                times+=1

        print("output1: \n", output1)
        # if is_unable_to_answer(output_all, chat_gm):
        #     output_all = PromptGenerate.final_answer(input_text[0],prompt, chat_gm)
        # print(output1)
        # time.sleep(60)
        ### --- End of Step 9 


        ### --- 10. Save the output
        ### --- save for json
        # json_data.append({
        #     "Q_ID": result[0]["Q_ID"],
        #     "question": result[0]["question"],
        #     "query_nodes": result[0]["query_nodes"],
        #     # "subgraph_dict": result[0]["subgraph_dict"],
        #     "paths-list": result[0]["paths-list"],
        #     "output": output1
        # })

        # # # 每 SAVE_INTERVAL 筆就存檔一次
        # # if (count + 1) % SAVE_INTERVAL == 0:
        # #     with open("./output/CommunitySearch/Final_Q714/GreedyGen/BFS_Last.json", "w", encoding="utf-8") as f:
        # #         json.dump(json_data, f, ensure_ascii=False, indent=4)

        # # with open("./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/E2_Path_MM/20250527_PR_new.json", "w", encoding="utf-8") as f:
        # #     json.dump(json_data, f, ensure_ascii=False, indent=4)
        # ### ---

        # # with open('./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/KeyEntities_Order/20250524_PR_new.csv', 'a+', newline='') as f6:
        # #     writer = csv.writer(f6)
        # #     writer.writerow([result[0]["Q_ID"], result[0]["question"], output1])
        # #     f6.flush()
        # # with open('../output/node_amount.csv', 'a+', newline='') as f6:
        # #     writer = csv.writer(f6)
        # #     writer.writerow([round_count, input_text[0], greedy_node, greedydist_node, greedygen_node])
        # #     f6.flush()

        # # time.sleep(5)
        ## --- End of Step 10


# with open("./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/KeyEntities_Order/20250524_PR_new.json", "w", encoding="utf-8") as f:
#     json.dump(json_data, f, ensure_ascii=False, indent=4)

