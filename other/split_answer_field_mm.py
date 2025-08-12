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
    df = pd.read_csv(csv_path, usecols=['Q_ID', 'Answer_all'])
    
    # 創建新欄位
    df['Most Likely Disease(s)'] = ''
    df['Suggested Medical Test(s)'] = ''
    df['Recommended Medication(s)'] = ''
    
    # 處理每一行
    for idx, row in df.iterrows():
        if pd.notna(row.get('Answer_all')):
            text = row['Answer_all']
            # print(text)
            
            # 提取疾病信息
            disease_pattern = r'\s*\*\*?Most Likely Disease(?:s|\(s\))?:\*\*?\s*(.*?)(?=2.\s*\*\*?Recommended)'
            disease_pattern_2 = r'\s*\*\*?Most Likely Diagnosi(?:s|\(s\))?:\*\*?\s*(.*?)(?=2.\s*\*\*?Recommended)'
            disease_pattern_3 = r'\s*\*\*?Most Likely Diagnosi(?:s|\(s\))?:\*\*?\s*(.*?)(?=2.\s*\*\*?Tests)'
            disease_pattern_4 = r'\s*\*\*?Possible Diagnosi(?:s|\(s\))?:\*\*?\s*(.*?)(?=2.\s*\*\*?Tests)'
            disease_pattern_5 = r'\s*\*\*?Possible Diagnosi(?:s|\(s\))?:\*\*?\s*(.*?)(?=2.\s*\*\*?Recommended)'
            # disease_pattern = r'\*\*Most Likely Disease\(s\):\*\*\s*(.*?)(?=\n\s*\d+\.\s*\*\*Recommended|$)'
            # disease_pattern = r'1\.\s*\*\*Most Likely Disease(?:s|\(s\))?:\*\*\s*(.*?)(?=\n\n2\.)'
            disease_match = re.search(disease_pattern, text, re.DOTALL)
            if disease_match:
                df.at[idx, 'Most Likely Disease(s)'] = disease_match.group(1).strip()
            else:
                disease_match = re.search(disease_pattern_2, text, re.DOTALL)
                if disease_match:
                    df.at[idx, 'Most Likely Disease(s)'] = disease_match.group(1).strip()
                else:
                    disease_match = re.search(disease_pattern_3, text, re.DOTALL)
                    if disease_match:
                        df.at[idx, 'Most Likely Disease(s)'] = disease_match.group(1).strip()
                    else:
                        disease_match = re.search(disease_pattern_4, text, re.DOTALL)
                        if disease_match:
                            df.at[idx, 'Most Likely Disease(s)'] = disease_match.group(1).strip()
                        else:
                            disease_match = re.search(disease_pattern_5, text, re.DOTALL)
                            if disease_match:
                                df.at[idx, 'Most Likely Disease(s)'] = disease_match.group(1).strip()
                            else:
                                print(f"Disease Q{row['Q_ID']}: ",disease_match)


            test_pattern = r'2.\s*\*\*?Recommended (?:Medical )?Test(?:s|\(s\))?:\*\*?\s*(.*?)(?=3.\s*\*\*?Recommended)'
            test_pattern_2 = r'2.\s*\*\*?Recommended (?:Medical )?Test(?:s|\(s\))?:\*\*?\s*(.*?)(?=3.\s*\*\*?Suggested)'
            test_pattern_3 = r'2.\s*\*\*?Tests to Confirm Diagnosi(?:s|\(s\))?:\*\*?\s*(.*?)(?=3.\s*\*\*?Recommended)'
            test_pattern_4 = r'2.\s*\*\*?Recommended Medication(?:s|\(s\))?(?:\s*\([^)]+\))?:\*\*?\s*(.*?)(?=3.\s*\*\*?Suggested)'
            # test_pattern = r'3\.\s*\*\*Suggested Medical Test(?:s|\(s\))?:\*\*\s*(.*)'
            test_match = re.search(test_pattern, text, re.DOTALL)
            # print(test_match)
            if test_match:
                df.at[idx, 'Suggested Medical Test(s)'] = test_match.group(1).strip()
            else:
                test_match = re.search(test_pattern_2, text, re.DOTALL)
                if test_match:
                    df.at[idx, 'Suggested Medical Test(s)'] = test_match.group(1).strip()
                else:
                    test_match = re.search(test_pattern_3, text, re.DOTALL)
                    if test_match:
                        df.at[idx, 'Suggested Medical Test(s)'] = test_match.group(1).strip()
                    else:
                        test_match = re.search(test_pattern_4, text, re.DOTALL)
                        if test_match:
                            df.at[idx, 'Recommended Medication(s)'] = test_match.group(1).strip()
                        else:
                            print(f"Suggested Medical Test(s) Q{row['Q_ID']}: ",test_match)
            

            # 提取藥物信息
            med_pattern = r'3\.\s*\*\*?Recommended Medication(?:s|\(s\))?(?:\s*\([^)]+\))?:\*\*?\s*(.*?)(?=\n\n\*\*|\Z)'
            med_pattern_2 = r'3\.\s*\*\*?Suggested Medication(?:s|\(s\))?(?:\s*\([^)]+\))?:\*\*?\s*(.*?)(?=\n\n\*\*|\Z)'
            med_pattern_3 = r'3\.\s*\*\*?Suggested (?:Medical )?Test(?:s|\(s\))?:\*\*?\s*(.*?)(?=\n\n\*\*|\Z)'
            
            # med_pattern = r'2\.\s*\*\*Recommended Medication(?:s|\(s\))?:\*\*\s*(.*?)(?=\n\n3\.)'
            med_match = re.search(med_pattern, text, re.DOTALL)
            if med_match:
                df.at[idx, 'Recommended Medication(s)'] = med_match.group(1).strip()
            else:
                # # 嘗試更寬鬆的匹配模式
                # alt_med_pattern = r'\n\s*\d+\.\s*\*\*Recommended Medication?(?:\(s\))?:\*\*)(?=\n\n\*\*|\s*$)'
                # alt_match = re.search(alt_med_pattern, text, re.DOTALL)
                # if alt_match:
                #     df.at[idx, 'Recommended Medication(s)'] = alt_match.group(1).strip()
                # else:
                med_match = re.search(med_pattern_2, text, re.DOTALL)
                if med_match:
                    df.at[idx, 'Recommended Medication(s)'] = med_match.group(1).strip()
                else:
                    med_match = re.search(med_pattern_3, text, re.DOTALL)
                    if med_match:
                        df.at[idx, 'Suggested Medical Test(s)'] = med_match.group(1).strip()
                    else:
                        print(f"Recommended Medication(s) Q{row['Q_ID']}: ",med_match)
    
    # 保存為新的 CSV 文件
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"處理後的 CSV 已儲存至: {output_path}")
        return None
    
    return df


if __name__ == "__main__":
    # Replace with your input and output file paths
    input_file = "../output/MindMap/Final_Q714_Gemma3-1B/20250518.csv"
    output_file = "../output/MindMap/Final_Q714_Gemma3-1B/20250520_3fields_new.csv"
    
    split_answer_field(input_file, output_file)