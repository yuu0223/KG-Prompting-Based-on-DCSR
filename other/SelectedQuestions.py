### 用於隨機選題測試

import random
import pandas as pd


with open("./data/chatdoctor5k/NER_chatgpt.json", "r") as f:
    lines = f.readlines()
    ### 設置隨機種子
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    # 總行數
    total_lines = len(lines)
    # 使用隨機種子選取 100 個題號
    selected_indices = random.sample(range(total_lines), 100)
    for i in range(len(selected_indices)):
        selected_indices[i]+=1
        
    print(f"隨機選取的題號: {selected_indices}")


df = pd.read_csv("./output_MindMap_Gemini15Flash.csv")
# 篩選 q_id 在目標清單中的資料
filtered_df = df[df['Question_ID'].isin(selected_indices)]
# 將結果存到新文件
filtered_df.to_csv("output_MindMap_100.csv", index=False)