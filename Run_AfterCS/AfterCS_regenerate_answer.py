import csv
import json
from communitysearch import PromptGenerate
import os
from llm import gemini
from tqdm import tqdm
from time import sleep

# 初始化 Gemini
GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")
chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)

# 讀取 paths JSON（你提到的 20250505.json）
with open("../output/SteinerTree/Final_Q714_Gemini1.5/20250505_PR_Gemini15.json", "r", encoding='utf-8') as f_json:
    path_data = json.load(f_json)

# 讀取原始 csv，準備儲存新結果
with open('../output/SteinerTree/Final_Q714_Gemma3-1B/20250519_PR_3.csv', 'r', encoding='utf-8') as f_csv, \
     open('../output/SteinerTree/Final_Q714_Gemma3-1B/20250519_PR_4.csv', 'w', newline='', encoding='utf-8') as f_out:

    reader = csv.DictReader(f_csv)
    writer = csv.DictWriter(f_out, fieldnames=['Q_ID', 'Question', 'Reference_Answer', 'Answer'])
    writer.writeheader()

    for row in tqdm(reader, desc="Reprocessing Empty Answers"):
        qid = row['Q_ID']
        question = row['Question']
        reference_answer = row['Reference_Answer']
        current_answer = row['Answer'].strip()

        # 跳過已經有答案的
        if current_answer and current_answer != "[]":
            writer.writerow(row)
            continue
    
        # 找對應的 path info
        print("Question: ", qid)
        result = [item for item in path_data if str(item.get("Q_ID")) == str(qid)]
        if not result:
            print(f"Q_ID={qid} not found in paths.")
            row['Answer'] = "Path not found."
            writer.writerow(row)
            continue

        # 準備 path prompt
        paths = result[0].get("paths-list", [])
        path_join = "There's no any reference path in this case." if not paths else "\n\n\n".join(paths)

        prompt = PromptGenerate.GeneratePathPrompt(path_input=path_join, chat_gm=chat_gm)

        # 呼叫 Gemini 回答（最多嘗試 2 次）
        times = 1
        while True:
            output_all = PromptGenerate.final_answer(question=question, prompt_path=prompt, chat_gm=chat_gm)
            print("Output: \n", output_all)
            output1_list = PromptGenerate.extract_final_answer(output_all)

            if output1_list:
                answer_generated = output1_list[0]
                break
            elif times == 2:
                answer_generated = []
                break
            else:
                sleep(3)
                times += 1

        # 寫入新答案
        row['Answer'] = answer_generated
        writer.writerow(row)
        f_out.flush()
