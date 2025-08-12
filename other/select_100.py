import random
import csv
import pandas as pd

df = pd.read_csv("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/MindMap/Final_Q714/20250408_MM_Gemini20Flash.csv")

with open("../data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        lines = f.readlines()
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        # 總行數
        total_lines = len(lines)
        # 使用隨機種子選取 100 個題號
        selected_indices = random.sample(range(total_lines), 718)
        # exclude = {552, 2553, 4333, 5084}  # 這裡是題號（從 1 開始）
        # selected_indices = [q for q in selected_indices if q not in exclude]

        filter_df = df[df["Question_ID"].isin(selected_indices)]

        filter_df.to_csv("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/MindMap/Final_Q714/MM_100.csv", index = False)