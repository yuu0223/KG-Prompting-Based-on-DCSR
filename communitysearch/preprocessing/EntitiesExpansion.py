### 這份檔案是為了看 Questions 和 Answers 所提取出來的 key entities 在 KG 中的 entities 為何
### 有另存檔案，Output csv

import pickle
import json
import re
import pandas as pd
import numpy as np
import csv
from neo4j import GraphDatabase
import os

uri = os.getenv("neo4j_uri")
username = os.getenv("neo4j_username")
password = os.getenv("neo4j_password")

driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()


def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim


def count_label_from_neo4j(match_kg):
    query_label = (
        f"WITH {match_kg} AS nodeNames"
        " MATCH (n)"
        " WHERE n.name IN nodeNames"
        " RETURN n.name AS node, labels(n) AS labels")
    
    result = session.run(query=query_label)

    labels = []
    for record in result:
        print(record)
        labels.append(record['labels'][1]) ### ['Entity', 'Symptom']
    
    unique_items = set(labels)
    counts = {item: labels.count(item) for item in unique_items}

    return counts


re1 = r'The extracted entities are (.*?)<END>'
re2 = r"The extracted entity is (.*?)<END>"
re3 = r"<CLS>(.*?)<SEP>"


with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
    entity_embeddings = pickle.load(f1)

    
with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/data/chatdoctor5k/keyword_embeddings_new.pkl','rb') as f2:
    keyword_embeddings = pickle.load(f2)


with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/data/preprocessing/entities_from_kg.csv', 'w', newline='') as f4:
    writer = csv.writer(f4)
    writer.writerow(['Index', 'Question', 'Q_KeyEntities', 'Q_KGEntities', 
                     'Answer', 'A_KeyEntities', 'A_KGEntities'])


with open("/home/yuu0223/KG-Prompting-Based-on-Community-Search/data/chatdoctor5k/NER_chatgpt.json", "r") as f:
    round_count=30
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
        print(f'Question{round_count}:\n',input_text[0])

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
        # print(answer_kg)

        
        match_question_kg = []
        match_answer_kg = []
        entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])

        ### Question kg
        for kg_entity in question_kg:
            print("Q's: ",kg_entity)
            keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
            max_index = cos_similarities.argmax()
                        
            match_kg_i = entity_embeddings["entities"][max_index]
            while match_kg_i.replace(" ","_") in match_question_kg:
                cos_similarities[max_index] = 0
                max_index = cos_similarities.argmax()
                match_kg_i = entity_embeddings["entities"][max_index]

            match_question_kg.append(match_kg_i.replace(" ","_"))
        
        ### Answer kg
        for kg_entity in answer_kg:
            print("A's: ",kg_entity)
            keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
            max_index = cos_similarities.argmax()
                        
            match_kg_i = entity_embeddings["entities"][max_index]
            while match_kg_i.replace(" ","_") in match_answer_kg:
                cos_similarities[max_index] = 0
                max_index = cos_similarities.argmax()
                match_kg_i = entity_embeddings["entities"][max_index]

            match_answer_kg.append(match_kg_i.replace(" ","_"))
        


        ### write in csv
        if output_text:
            with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/data/preprocessing/entities_from_kg.csv', 'a+', newline='') as f6:
                writer = csv.writer(f6)
                writer.writerow([round_count, input_text[0], question_kg, match_question_kg,
                                output_text[0], answer_kg, match_answer_kg])
                f6.flush()
        else:
            with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/data/preprocessing/entities_from_kg.csv', 'a+', newline='') as f6:
                writer = csv.writer(f6)
                writer.writerow([round_count, input_text[0], question_kg, match_question_kg,
                                "No answer output.", [], []])
                f6.flush()