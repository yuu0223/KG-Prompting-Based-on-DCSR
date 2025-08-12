### --- 這份檔案是給執行完CommunitySearch找出路徑後的Gemini產生回答階段
### 1086

import json
from communitysearch import PromptGenerate
import os
import csv
import re
from llm import gemini
import random
from tqdm import tqdm
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from mindmap import FinalPrompt
import time


GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade_3")
chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)


def prompt_path_finding(path_input):
    template = """
    There are some knowledge graph path. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                        text={})

    # response_of_KG_path = chat(chat_prompt_with_values.to_messages()).content
    response_of_KG_path = chat_gm(chat_prompt_with_values.to_messages()[0].content)
    
    return response_of_KG_path


def prompt_neighbor(neighbor):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text={})

    # response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content
    response_of_KG_neighbor = chat_gm(chat_prompt_with_values.to_messages()[0].content)
    

    return response_of_KG_neighbor


def extract_final_answer(output_all):
    # re4 = r"Output ?1:? ?(.*?) ?Output ?2:?"
    re4 = r"Output\s*1\s*:\s*(.*?)\s*Output\s*2\s*:"
    re4_2 = r"Output\s*1\s*:\s*(.*)"
    # re5 = r"Output 2:(.*?)Output 3:"

    ### Summary
    output1 = re.findall(re4, output_all, flags=re.DOTALL)
    if output1 == []:
        output1 = re.findall(re4_2, output_all, flags=re.DOTALL)
    # re5 = r"Output 2:(.*?)Output 3:"
    
    # ### Path Evidence
    # output2 = re.findall(re5, output_all, flags=re.DOTALL)
    # if len(output2) > 0:
    #     output2 = output2[0]
    # else:
    #     continue
    
    # ### Decision Tree
    # output3_index = output_all.find("Output 3:")
    # if output3_index != -1:
    #     output3 = output_all[output3_index + len("Output 3:"):].strip()
    
    # output_community = output1
    # print("community search: \n", output1)
    return output1


#### ---- Start!!!


# with open('./output/MindMap/Final_Q714_Gemma3-4B/20250513.csv', 'w', newline='') as f4: ###
#     writer = csv.writer(f4)
#     writer.writerow(['Q_ID', 'Question', 'Reference_Answer', 'Answer_all', 'Answer_path', 'Answer_neighbor'])

re3 = r"<CLS>(.*?)<SEP>"

json_data = []
# SAVE_INTERVAL = 5
# with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
#     round_count = 0 ## Initial : 30
#     for line in f.readlines()[round_count:]:
#         round_count+=1
#         x = json.loads(line)
#         answer_output = x["answer_output"]
#         print("Q_ID from data:", x["Q_ID"])

with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

round_count = 338

for line in tqdm(lines[round_count:round_count+1], desc="Processing Q&A", unit="question", dynamic_ncols=True):
    x = json.loads(line)
    answer_output = x["answer_output"]
    print("Q_ID from data:", x["Q_ID"])

    with open("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/MindMap/Final_Q714_Gemini1.5/20250505_Gemini15.json", "r", encoding='utf-8') as f1: ###
        data = json.load(f1)
        id = x["Q_ID"]
        result = [item for item in data if item.get("Q_ID") == id]
        print("Q_ID from paths json:", result[0]["Q_ID"])
        # print("result:", result)
        # print(result[0]["paths-list"]) if result else print(f"Q_ID={id} is not found.")
        

        ### --- 7. 將 Paths 轉換為 Natural Language
        if result :
            if result[0]["path-based"] == []:
                response_of_KG_list_path = "There's no any reference path in this case."
            else:
                path = "\n".join(result[0]["path-based"])
                print("path:", path)
                response_of_KG_list_path = prompt_path_finding(path)
                print("response_of_KG_list_path", response_of_KG_list_path)
            
            if result[0]["neighbor-based"] == []:
                response_of_KG_neighbor = "There's no any reference path in this case."
            else:
                neighbor_input = "\n".join(result[0]["neighbor-based"])
                response_of_KG_neighbor = prompt_neighbor(neighbor_input)

        else:
            print(f"Q_ID={id} is not found.")
            continue
        ### --- End of Step 7


        ### --- 8. KG-based Prompt Generation
        # --- ALL
        times=1
        while True:
            output_all = FinalPrompt.final_answer_for_orginal(result[0]["question"],response_of_KG_list_path,response_of_KG_neighbor, chat_gm)
            # print("output_all: \n", output_all)
            output1 = extract_final_answer(output_all)
            # output1, output2, output3 = extract_final_answer(output_all)
            if len(output1) > 0:
                output1 = output1[0]
                # output2 = output2[0]
                # output3 = output3[0]
                break
            elif times == 2:
                print("second round.")
                break
            else:
                times+=1

        print("output1: \n", output1)
        
        # time.sleep(5)

        # # --- Path-based
        # times=1
        # while True:
        #     output_all_path = FinalPrompt.final_answer_for_orginal(result[0]["question"],response_of_KG_list_path,"", chat_gm)
        #     # print("output_all: \n", output_all)
        #     output_path = extract_final_answer(output_all_path)
        #     # output1, output2, output3 = extract_final_answer(output_all)
        #     if len(output_path) > 0:
        #         output_path = output_path[0]
        #         # output2 = output2[0]
        #         # output3 = output3[0]
        #         break
        #     elif times == 2:
        #         print("second round.")
        #         break
        #     else:
        #         times+=1

        # print("output_path: \n", output_path)

        # time.sleep(5)

        # # --- Neighbor-based
        # times=1
        # while True:
        #     output_all_neighbor = FinalPrompt.final_answer_for_orginal(result[0]["question"],"",response_of_KG_neighbor, chat_gm)
        #     # print("output_all: \n", output_all)
        #     output_neighbor = extract_final_answer(output_all_neighbor)
        #     # output1, output2, output3 = extract_final_answer(output_all)
        #     if len(output_neighbor) > 0:
        #         output_neighbor = output_neighbor[0]
        #         # output2 = output2[0]
        #         # output3 = output3[0]
        #         break
        #     elif times == 2:
        #         print("second round.")
        #         break
        #     else:
        #         times+=1

        # print("output_neighbor: \n", output_neighbor)

        ## --- End of Step 9 


        ### --- 10. Save the output
#         ### --- save for json
#         json_data.append({
#             "Q_ID": result[0]["Q_ID"],
#             "question": result[0]["question"],
#             "query_nodes": result[0]["query_nodes"],
#             "path-based": result[0]["path-based"],
#             "neighbor-based": result[0]["neighbor-based"],
#             "output_all": output1
#         })

#         # # 每 SAVE_INTERVAL 筆就存檔一次
#         # if (count + 1) % SAVE_INTERVAL == 0:
#         #     with open("./output/CommunitySearch/Final_Q714/GreedyGen/BFS_Last.json", "w", encoding="utf-8") as f:
#         #         json.dump(json_data, f, ensure_ascii=False, indent=4)

#         # with open("./output/MindMap/E1_KeyEntities_Order_NodeDegree/20250527_new.json", "w", encoding="utf-8") as f:
#         #     json.dump(json_data, f, ensure_ascii=False, indent=4)
#         ## ---

#         with open('./output/MindMap/E1_KeyEntities_Order_NodeDegree/20250527.csv', 'a+', newline='') as f6:
#             writer = csv.writer(f6)
#             writer.writerow([result[0]["Q_ID"], result[0]["question"], output1])
#             f6.flush()
#         # with open('./output/node_amount.csv', 'a+', newline='') as f6:
#         #     writer = csv.writer(f6)
#         #     writer.writerow([round_count, input_text[0], greedy_node, greedydist_node, greedygen_node])
#         #     f6.flush()

#         # time.sleep(5)
#         ## --- End of Step 10


# with open("./output/MindMap/E1_KeyEntities_Order_NodeDegree/20250527_new.json", "w", encoding="utf-8") as f:
#     json.dump(json_data, f, ensure_ascii=False, indent=4)

