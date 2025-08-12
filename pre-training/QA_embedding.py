import re
import json
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

re1 = r'The extracted entities are (.*?)<END>'
re2 = r"The extracted entity is (.*?)<END>"
re3 = r"<CLS>(.*?)<SEP>"

keywords_set = set()
with open("/home/yuu0223/MindMap_Test/data/chatdoctor5k/NER_chatgpt.json", "r") as f:
    round_count=0
    for line in f.readlines()[round_count:]:
        x = json.loads(line)
        input = x["qustion_output"]
        input = input.replace("\n","")
        input = input.replace("<OOS>","<EOS>")
        input = input.replace(":","") + "<END>"
        input_text = re.findall(re3,input)
        
        if input_text == []:
            continue
        round_count+=1
        print('Question',round_count,':\n',input_text[0])

        output = x["answer_output"]
        output = output.replace("\n","")
        output = output.replace("<OOS>","<EOS>")
        output = output.replace(":","") + "<END>"
        output_text = re.findall(re3,output)
        # print(output_text[0])

                
        question_kg = re.findall(re1,input)
        if len(question_kg) == 0:
            question_kg = re.findall(re2,input)
            if len(question_kg) == 0:
                print("<Warning> no entities found", input)
                continue
        question_kg = question_kg[0].replace("<END>","").replace("<EOS>","")
        question_kg = question_kg.replace("\n","")
        question_kg = question_kg.split(", ")
        # print("question_kg",question_kg)

        answer_kg = re.findall(re1,output)
        if len(answer_kg) == 0:
            answer_kg = re.findall(re2,output)
            if len(answer_kg) == 0:
                print("<Warning> no entities found", output)
                continue
        answer_kg = answer_kg[0].replace("<END>","").replace("<EOS>","")
        answer_kg = answer_kg.replace("\n","")
        answer_kg = answer_kg.split(", ")

        [keywords_set.add(q_keyword) for q_keyword in question_kg]
        [keywords_set.add(a_keyword) for a_keyword in answer_kg]
        # time.sleep(10)
    
keywords_list = list(keywords_set)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model.to("cuda")

# encode keywords
embeddings = model.encode(keywords_list, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
keyword_emb_dict = {
    "keywords": keywords_list,
    "embeddings": embeddings,
}
import pickle
with open("/home/yuu0223/MindMap_Test/data/chatdoctor5k/keyword_embeddings_new.pkl", "wb") as f:
    pickle.dump(keyword_emb_dict, f)

print("Keywords Embedding Done!")
