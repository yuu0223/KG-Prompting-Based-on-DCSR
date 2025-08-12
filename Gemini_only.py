import json
import os
import csv
import re
from llm import gemini
from tqdm import tqdm
from datetime import date
import time


if __name__ == "__main__":

    ### --- 1. Gemini API Connection
    GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of step 1


    ### --- 2. Question Processing & Gemini Answering
    with open(f'./output/Gemini/Final_Q714_Gemma3-1B/{date.today()}.csv', 'w', newline='') as f4: ###
        writer = csv.writer(f4)
        writer.writerow(['Q_ID', 'Question', 'Reference Answer', 'Answer'])

    re3 = r"<CLS>(.*?)<SEP>"

    with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
        lines = f.readlines()

    round_count = 0
    for line in tqdm(lines[round_count:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
        x = json.loads(line)
        print("Q_ID from data:", x["Q_ID"])

        input = x["question_output"]
        input = input.replace("\n","")
        input = input.replace("<OOS>","<EOS>")
        input = input.replace(":","") + "<END>"
        input_text = re.findall(re3,input)
        question = input_text[0]
        print('Question:', x["Q_ID"],'\n',question)

        answer = x["answer_output"]
            
        prompt = (
            f"Patient input: {question}\n\n"
            "What disease does the patient have? What tests should patient take to confirm the diagnosis? "
            "What recommened medications can cure the disease? Think step by step.\n\n\n"
            "Answer in the following format with three clearly separated sections:\n"
            "1. **Most Likely Disease:**\n[State only one most likely disease based on the symptoms. Briefly explain why.]\n\n"
            "2. **Recommended Medication(s):**\n[List multiple possible medications or treatment options for this disease, if applicable.]\n\n"
            "3. **Suggested Medical Test(s):**\n[List multiple relevant medical tests that can help confirm or rule out the diagnosis.]\n\n"
        )
        
        output_all = chat_gm(prompt)
        
        with open(f'./output/Gemini/Final_Q714_Gemma3-1B/{date.today()}.csv', 'a+', newline='') as f6:
            writer = csv.writer(f6)
            writer.writerow([x["Q_ID"], question, answer, output_all])
            f6.flush()
    ### --- End of step 2