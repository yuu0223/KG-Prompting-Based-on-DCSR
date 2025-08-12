import pandas as pd
from collections import defaultdict

# 讀取 CSV
df = pd.read_csv("/home/yuu0223/KG-Prompting-Based-on-Community-Search/evaluation/Compare/drug_20250401_1vs1.csv")

# 初始化分數
# score = defaultdict(float)
win = defaultdict(float)
tie = defaultdict(float)

# 計分邏輯
for _, row in df.iterrows():
    first = row['first_output']
    second = row['second_output']
    result = row['compare_result']

    if result == 1:
        win[first] += 1
    elif result == 2:
        win[second] += 1
    elif result == 0:
        tie[first] += 1
        tie[second] += 1

# 計算每個 output 的出場次數（每出場一次就算一場）
game_count = defaultdict(int)
for _, row in df.iterrows():
    game_count[row['first_output']] += 1
    game_count[row['second_output']] += 1

# 計算勝率
win_rate = {output: round(win[output] / game_count[output], 3) for output in win}
tie_rate = {output: round(tie[output] / game_count[output], 3) for output in tie}

# 顯示結果
print("win:")
for output, rate in win_rate.items():
    print(f"{output}: win rate = {rate} ({win[output]} pts / {game_count[output]} games)")
print("tie:")
for output, rate in tie_rate.items():
    print(f"{output}: tie rate = {rate} ({tie[output]} pts / {game_count[output]} games)")
