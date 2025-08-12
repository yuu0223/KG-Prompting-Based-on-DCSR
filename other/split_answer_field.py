import pandas as pd
import re

def split_answer_field(csv_path, output_path=None):
    """
    將 CSV 文件中的 'answer' 欄位分割成三個獨立的欄位:
    1. Most Likely Disease
    2. Recommended Medication(s)
    3. Suggested Medical Test(s)
    
    Parameters:
    -----------
    csv_path : str
        輸入 CSV 文件的路徑
    output_path : str, optional
        輸出 CSV 文件的路徑。如果為 None，則返回處理後的 DataFrame
        
    Returns:
    --------
    pandas.DataFrame 或 None
        如果 output_path 為 None，則返回處理後的 DataFrame
    """
    # 讀取 CSV 文件
    df = pd.read_csv(csv_path, usecols=['Q_ID', 'Answer'])
    
    # 創建新欄位
    df['Most Likely Disease'] = ''
    df['Recommended Medication(s)'] = ''
    df['Suggested Medical Test(s)'] = ''
    
    # 處理每一行
    for idx, row in df.iterrows():
        if pd.notna(row.get('Answer')):
            text = row['Answer']
            # print(text)
            
            # 提取疾病信息
            disease_pattern = r'\*\*Most Likely Disease:\*\*\s*(.*?)(?=2.\s\*\*Recommended Medication(?:s|\(s\)|s\(|\(s|\(s:|s\))?:\*\*\s*(.*?))'
            # disease_pattern = r'1\.\s*\*\*Most Likely Disease(?:s|\(s\))?:\*\*\s*(.*?)(?=\n\n2\.)'
            disease_match = re.search(disease_pattern, text, re.DOTALL)
            if disease_match:
                df.at[idx, 'Most Likely Disease'] = disease_match.group(1).strip()
            else:
                print(f"Disease Q{row['Q_ID']}: ",disease_match)
            
            # 提取藥物信息
            med_pattern = r'\*\*Recommended Medication\(s\):\*\*\s*(.*?)(?=3.\s\*\*Suggested Medical Test(?:s|\(s\)|s\(|\(s|\(s:|s\))?:\*\*\s*(.*?))'
            # med_pattern = r'2\.\s*\*\*Recommended Medication(?:s|\(s\))?:\*\*\s*(.*?)(?=\n\n3\.)'
            med_match = re.search(med_pattern, text, re.DOTALL)
            if med_match:
                df.at[idx, 'Recommended Medication(s)'] = med_match.group(1).strip()
            else:
                print(f"Medication Q{row['Q_ID']}: ",med_match)
            
            # 提取測試信息
            test_pattern = r'\*\*Suggested Medical Test(?:s|\(s\)|s\(|\(s|\(s:|s\))?:\*\*\s*(.*?)(?=\n\n\d+\.|\n\n\*\*|\s*$)'
            # test_pattern = r'3\.\s*\*\*Suggested Medical Test(?:s|\(s\))?:\*\*\s*(.*)'
            test_match = re.search(test_pattern, text, re.DOTALL)
            # print(test_match)
            if test_match:
                df.at[idx, 'Suggested Medical Test(s)'] = test_match.group(1).strip()
            else:
                # 嘗試更寬鬆的匹配模式
                alt_test_pattern = r'3\.\s*\*\*Suggested Medical Test\(s\):\*\*\s*(.*?)(?=\n\n\*\*|\s*$)'
                alt_match = re.search(alt_test_pattern, text, re.DOTALL)
                if alt_match:
                    df.at[idx, 'Suggested Medical Test(s)'] = alt_match.group(1).strip()
                else:
                    print(f"Suggested Medical Test(s) Q{row['Q_ID']}: ",alt_match)
    
    # 保存為新的 CSV 文件
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"處理後的 CSV 已儲存至: {output_path}")
        return None
    
    return df


if __name__ == "__main__":
    # Replace with your input and output file paths
    input_file = "../output/SteinerTree/Final_Q714_Gemma3-1B/20250519_PR.csv"
    output_file = "../output/SteinerTree/Final_Q714_Gemma3-1B/20250520_3fields_PR.csv"
    
    split_answer_field(input_file, output_file)