import openai
import csv
from time import sleep
import os

openai.api_key = os.getenv("openai_api_key")

def compare_prompt(reference_ans, first_output, second_output):
    template = """
    reference answer: {reference_ans}
    \n\n
    output1: {first_output}
    \n\n
    output2: {second_output}
    \n\n
    Check if the reference answer mentions a tests recommendation result:
        1. If yes, choose the output (1 or 2) that best matches it.  
        2. If not, use your medical knowledge to decide which output is closer to the correct tests recommendation result.

    Return only one number:  
        - "1" if output1 is better.  
        - "2" if output2 is better.  
        - "0" if both are similar.  
    Do not explain.
    """

    prompt = template.format(reference_ans=reference_ans, first_output=first_output, second_output=second_output)

    # 将 prompt 作为输入传递给 GPT-4 模型进行生成
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
          {"role": "system", "content": "You are an excellent AI doctor."},
          {"role": "user", "content": prompt}
          ]) 
    response_of_comparation = response.choices[0].message.content

    return response_of_comparation


input_file = '/home/yuu0223/KG-Prompting-Based-on-Community-Search/evaluation/1_merge_file/merged_output_100_20250324_new.csv'
output_file = '/home/yuu0223/KG-Prompting-Based-on-Community-Search/evaluation/Compare/test_20250401.csv'


with open(output_file, 'w', newline='') as f1:
    writer = csv.writer(f1)
    # writer.writerow(['Question_ID', 'first_output', 'second_output', 'compare_result'])
    
    start_line = 87  # 從第 100 行開始
    line_number = 0

    with open(input_file,'r',newline="") as f_input, open(output_file, 'a+', newline='') as f_output:
        reader = csv.reader(f_input)
        header = next(reader)

        for row in reader:
            line_number += 1
            if line_number < start_line:
                continue  # 跳過直到第 100 行

            for row in reader:
                output1_text = [row[3].strip("\n")]
                output2_text = [row[4].strip("\n")]
                output3_text = [row[5].strip("\n")]
                output4_text = [row[6].strip("\n")]
                output5_text = [row[7].strip("\n")]
                output6_text = [row[8].strip("\n")]
                output7_text = [row[9].strip("\n")]
                
                Q_id = row[0].strip("\n")
                references = [row[2].strip("\n")]

                output_var = ["tab", output1_text, output2_text, output3_text, 
                            output4_text, output5_text, output6_text, output7_text]
                
                output_text = ["tab", "output1", "output2", "output3", 
                            "output4", "output5", "output6", "output7"]

                for i in range(1,8,1):
                    for j in range(i+1,8,1):
                        response = compare_prompt(references, output_var[i], output_var[j])
                        writer.writerow([Q_id, output_text[i], output_text[j], response])
                        f1.flush()

                        sleep(1)
