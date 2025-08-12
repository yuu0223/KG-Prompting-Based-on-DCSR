import pandas as pd
import json

### --- GPT Ranking
# df  = pd.read_csv("/home/yuu0223/KG-Prompting-Based-on-Community-Search/evaluation/3_Ranking_Compute/output_ranking_compute_Q714_Gemini15Flash_0506.csv")
### --- BERTScore
df  = pd.read_csv("/home/yuu0223/KG-Prompting-Based-on-Community-Search/evaluation/BERTscore/Q714_Gemini1.5/0506_Q714_Gemini15_new.csv")


df_index  = pd.read_csv("/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/cal_QN_shortest_distance_final.csv")
### --- Level A (Average : 1 ~ 2)
qids = df_index[df_index["Level"] == "A"]["Q_ID"].tolist()
### --- Level B (Average : 2 ~ 4)
# qids = df_index[df_index["Level"] == "B"]["Q_ID"].tolist()
### --- Level C (Average : 4 ~ )
# qids = df_index[df_index["Level"] == "C"]["Q_ID"].tolist()
### --- Level Z (Average : 0 )
# qids = df_index[df_index["Level"] == "Z"]["Q_ID"].tolist()


filtered_df = df[df['Q_ID'].isin(qids)]
filtered_df.to_csv("/home/yuu0223/KG-Prompting-Based-on-Community-Search/evaluation/BERTscore/Q714_Gemini1.5/QN_distance/Level_A.csv", index=False)