### --- 這份檔案是給執行完CommunitySearch找出路徑後的Gemini產生回答階段
### 1086

import json
import csv
from tqdm import tqdm


with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
    lines = f.readlines()

with open("./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/E5_GreedyDist_300/20250626_PR.json", "r", encoding='utf-8') as f1: ###
    data = json.load(f1)

round_count = 0


for line in tqdm(lines[round_count:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
    x = json.loads(line)
    answer_output = x["answer_output"]
    print("Q_ID from data:", x["Q_ID"])

    id = x["Q_ID"]
    result = [item for item in data if item.get("Q_ID") == id]
    print("Q_ID from paths json:", result[0]["Q_ID"])
    
    if result :
        with open('./output/CommunitySearch/Final_Q714_Gemini1.5/GreedyDist/E5_GreedyDist_300/20250626_PR.csv', 'a+', newline='') as f6:
            writer = csv.writer(f6)
            writer.writerow([result[0]["Q_ID"], result[0]["question"], result[0]["output"]])
            f6.flush()
    else:
        print(f"Q_ID={id} is not found.")
        continue



