### --- 這份檔案是給執行完CommunitySearch找出路徑後的Gemini產生回答階段

import json
from communitysearch import PromptGenerate
import os
import csv
import re
from llm import gemini
import random
from tqdm import tqdm

### --- 2. OpenAI API based keyword extraction and match entities
GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")
chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
### --- End of Step 2

re3 = r"<CLS>(.*?)<SEP>"

json_data = []
SAVE_INTERVAL = 5
with open("../data/chatdoctor5k/NER_chatgpt.json", "r") as f:
    lines = f.readlines()
    ### 設置隨機種子
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    # 總行數
    total_lines = len(lines)
    # 使用隨機種子選取 100 個題號
    # selected_indices = random.sample(range(total_lines), 718)
    # exclude = {552, 2553, 4333, 5084}  # 這裡是題號（從 1 開始）
    # selected_indices = [q for q in selected_indices if q not in exclude]
    # selected_indices = selected_indices[428:]

    selected_indices = [17, 2281]
    print(f"隨機選取的題號: {selected_indices}")

    # for i in range(45, len(selected_indices)+1): 
    #     index = selected_indices[i]
    for count, index in enumerate(tqdm(selected_indices, desc="Processing Questions", unit="question", dynamic_ncols=True)):
        try: 
            line = lines[index]
            x = json.loads(line)

            output = x["answer_output"]
            output = output.replace("\n","")
            output = output.replace("<OOS>","<EOS>")
            output = output.replace(":","") + "<END>"
            output_text = re.findall(re3,output)

            with open("../output/SteinerTree/Final_Q714/20250409_BFS_Gemini20Flash.json", "r", encoding='utf-8') as f:
                data = json.load(f)
                id = index+1
                result = [item for item in data if item.get("Q_ID") == id]
                # print("result:", result)

                # print(result[0]["paths-list"]) if result else print(f"Q_ID={id} is not found.")
                
                if result :
                    if result[0]["paths-list"] == []:
                        path_join = "There's no any reference path in this case."
                    else:
                        paths = result[0]["paths-list"]
                        path_join = "\n\n\n".join(paths)
                else:
                    print(f"Q_ID={id} is not found.")
                    continue
                

                ### --- 7. 將 Paths 轉換為 Natural Language
                prompt = PromptGenerate.GeneratePathPrompt(path_join, chat_gm)
                ### --- End of Step 7

                ### --- 9. KG-based Prompt Generation
                # 若沒有回答出來，讓 LLM 重新回答
                times=1
                while True:
                    output_all = PromptGenerate.final_answer(result[0]["question"], prompt, chat_gm)
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
                    "Q_ID": result[0]["Q_ID"],
                    "question": result[0]["question"],
                    "query_nodes": result[0]["query_nodes"],
                    "subgraph_dict": result[0]["subgraph_dict"],
                    "paths-list": result[0]["paths-list"],
                    "output": output1
                })

                # # 每 SAVE_INTERVAL 筆就存檔一次
                # if (count + 1) % SAVE_INTERVAL == 0:
                #     with open("../output/CommunitySearch/Final_Q714/GreedyGen/BFS_Last.json", "w", encoding="utf-8") as f:
                #         json.dump(json_data, f, ensure_ascii=False, indent=4)

                with open("../output/SteinerTree/Final_Q714/BFS_Last.json", "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
                ### ---

                with open('../output/SteinerTree/Final_Q714/20250409_BFS_Gemini20Flash.csv', 'a+', newline='') as f6:
                    writer = csv.writer(f6)
                    writer.writerow([result[0]["Q_ID"], result[0]["question"], output_text[0], output1])
                    f6.flush()
                # with open('../output/node_amount.csv', 'a+', newline='') as f6:
                #     writer = csv.writer(f6)
                #     writer.writerow([round_count, input_text[0], greedy_node, greedydist_node, greedygen_node])
                #     f6.flush()

                # time.sleep(5)
                ### --- End of Step 10
        
        except Exception as e:
            print(f"錯誤在第 {index+1} 題: {e}")


with open("../output/SteinerTree/Final_Q714/BFS_Last.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

