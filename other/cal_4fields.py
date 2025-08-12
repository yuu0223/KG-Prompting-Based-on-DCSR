from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
import openai
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from langchain.llms import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from time import sleep

import BuildDatabase
from llm import gemini

from tqdm import tqdm
import random
from mindmap import FinalPrompt


def count_label_from_neo4j(match_kg):
    query_label = (
        f"WITH {match_kg} AS nodeNames"
        " MATCH (n)"
        " WHERE n.name IN nodeNames"
        " RETURN n.name AS node, labels(n) AS labels"
    )
    result = session.run(query=query_label)

    labels = []
    for record in result:
        # print(record)
        labels.append(record['labels'][1]) ### ['Entity', 'Symptom']
    
    unique_items = set(labels)
    counts = {item: labels.count(item) for item in unique_items}

    return counts


# def chat_35(prompt):
#     completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#     {"role": "user", "content": prompt}
#     ])
#     return completion.choices[0].message.content

# def chat_4(prompt):
#     completion = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#     {"role": "user", "content": prompt}
#     ])
#     return completion.choices[0].message.content


def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    global exist_entity
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)
           
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")
                    path_str += "->" + relations[i] + "->"
            
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}
            
        if len(paths) > 5:        
            paths = sorted(paths, key=len)[:5]

        return paths,exist_entity


def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results


def get_entity_neighbors(entity_name: str,disease_flag) -> List[List[str]]:
    disease = []
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]
        
        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]
        
        if "disease" in rel_type.replace("_"," "):
            disease.extend(neighbors)

        else:
            neighbor_list.append([entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in neighbors])
                                ])
    
    return neighbor_list,disease

def prompt_path_finding(path_input):
    template = """
    There are some knowledge graph path. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                        text={})

    # response_of_KG_path = chat(chat_prompt_with_values.to_messages()).content
    response_of_KG_path = chat_gm(chat_prompt_with_values.to_messages()[0].content)
    
    return response_of_KG_path

def prompt_neighbor(neighbor):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text={})

    # response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content
    response_of_KG_neighbor = chat_gm(chat_prompt_with_values.to_messages()[0].content)
    

    return response_of_KG_neighbor

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

# def is_unable_to_answer(response):
 
#     analysis = openai.Completion.create(
#     engine="text-davinci-002",
#     prompt=response,
#     max_tokens=1,
#     temperature=0.0,
#     n=1,
#     stop=None,
#     presence_penalty=0.0,
#     frequency_penalty=0.0
# )
#     score = analysis.choices[0].text.strip().replace("'", "").replace(".", "")
#     if not score.isdigit():   
#         return True
#     threshold = 0.6
#     if float(score) > threshold:
#         return False
#     else:
#         return True


def autowrap_text(text, font, max_width):

    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

def final_answer(str,response_of_KG_list_path,response_of_KG_neighbor):
    messages  = [
                {"role": "system", "content":"You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. "},
                {"role": "user", "content":"Patient input:"+ input_text[0]},
                {"role": "assistant", "content":"You have some medical knowledge information in the following:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor},
                {"role": "user", "content":"What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? Think step by step.\n\n\n"
                            + "Output1: The answer includes disease and tests and recommened medications.\n\n"
                             +"Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighor-based Evidence, and in the end infer what result. \n Transport the inference process into the following format:\n Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...). \n\n"
                             +"Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n\n"
                             + "There is a sample:\n"
                             + """
Output 1:
Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.

Output 2:
Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')->result 1('laryngitis')->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').

Output 3: 
Patient(Path-based Evidence 1)
└── has been experiencing(Path-based Evidence 1)
    └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)
        └── could be caused by(Path-based Evidence 2)
            └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)
                ├── requires(Neighbor-based Evidence 1)
                │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)
                │       └── may include(Neighbor-based Evidence 2)
                │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)
                ├── can be treated with(Path-based Evidence 3)
                │   └── anti-inflammatory drugs and steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)
                └── should be accompanied by(Neighbor-based Evidence 3)
                    └── resting the voice and avoiding irritants(Neighbor-based Evidence 3)
                                    """
                             }]
        
    # result = chat(messages)
    # output_all = result.content
    prompt = "\n\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

    output_all = chat_gm(prompt)

    return output_all

def prompt_document(question,instruction):
    template = """
    You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.\n\n
    Patient input:\n
    {question}
    \n\n
    You have some medical knowledge information in the following:
    {instruction}
    \n\n
    What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease?
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    # response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content
    response_document_bm25 = chat_gm(chat_prompt_with_values.to_messages()[0]).content

    return response_document_bm25

def extract_final_answer(output_all):
    re4 = r"Output ?1:? ?(.*?) ?Output ?2:?"
    # re5 = r"Output 2:(.*?)Output 3:"

    ### Summary
    output1 = re.findall(re4, output_all, flags=re.DOTALL)
    
    # ### Path Evidence
    # output2 = re.findall(re5, output_all, flags=re.DOTALL)
    # if len(output2) > 0:
    #     output2 = output2[0]
    # else:
    #     continue
    
    # ### Decision Tree
    # output3_index = output_all.find("Output 3:")
    # if output3_index != -1:
    #     output3 = output_all[output3_index + len("Output 3:"):].strip()
    
    # output_community = output1
    # print("community search: \n", output1)
    return output1


if __name__ == "__main__":

    ### ---- 1. build neo4j knowledge graph datasets
    uri = os.getenv("neo4j_uri")
    username = os.getenv("neo4j_username")
    password = os.getenv("neo4j_password")
    print(uri) # 檢查用

    # --- build KG 
    # data_path = './data/chatdoctor5k/train.txt'
    # BuildDatabase.BuildNeo4j(data_path)
    # ---

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()
    ### --- End of Step 1
    

    print("Start to match the entites...")
    ### --- 2. OpenAI API based keyword extraction and match entities
    GEMINI_API_KEY = os.getenv("gemini_api_key")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of Step 2


# # 2. OpenAI API based keyword extraction and match entities

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    # with open('./output/MindMap/Final_Q714/20250408_MM_Gemini20Flash.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(['Question_ID', 'Question', 'Answer', 'MindMap'])


    with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open('./data/chatdoctor5k/keyword_embeddings_new.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)
    

    with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/entities_classification/data_classification_final.csv', 'w', newline='') as f3:
        writer = csv.writer(f3)
        writer.writerow(['Index', 'Question', 'Q_Disease', 'Q_Medical_Test', 'Q_Medication', 'Q_Symptom', 
                     'Answer', 'A_Disease', 'A_Medical_Test', 'A_Medication', 'A_Symptom'])
    
    with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/entities_classification/data_classification_percent_final.csv', 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Index', 'Question', 'Q_Disease', 'Q_Medical_Test', 'Q_Medication', 'Q_Symptom', 
                     'Answer', 'A_Disease', 'A_Medical_Test', 'A_Medication', 'A_Symptom'])

    docs_dir = './data/chatdoctor5k/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)
    

    json_data = []
    SAVE_INTERVAL = 5
    with open("./data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        lines = f.readlines()
        ### 設置隨機種子
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        # 總行數
        total_lines = len(lines)
        # 使用隨機種子選取 100 個題號
        selected_indices = random.sample(range(total_lines), 718)
        exclude = {552, 2553, 4333, 5084}  # 這裡是題號（從 1 開始）
        selected_indices = [q for q in selected_indices if q not in exclude]
        print(f"隨機選取的題號: {selected_indices}")

        # for i in range(552, len(selected_indices)+1): 
        #     index = selected_indices[i]
        for count, index in enumerate(tqdm(selected_indices, desc="Processing Questions", unit="question", dynamic_ncols=True)):
            try:
                line = lines[index]
                x = json.loads(line)
                input = x["qustion_output"]
                input = input.replace("\n","")
                input = input.replace("<OOS>","<EOS>")
                input = input.replace(":","") + "<END>"
                input_text = re.findall(re3,input)
                
                if input_text == []:
                    continue
                print('Question:', index+1,'\n',input_text[0])

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

                
                match_kg = []
                entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
            

                # for kg_entity in question_kg:
                    
                #     keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                #     kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                #     cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                #     max_index = cos_similarities.argmax()
                            
                #     match_kg_i = entity_embeddings["entities"][max_index]
                #     while match_kg_i.replace(" ","_") in match_kg:
                #         cos_similarities[max_index] = 0
                #         max_index = cos_similarities.argmax()
                #         match_kg_i = entity_embeddings["entities"][max_index]

                #     match_kg.append(match_kg_i.replace(" ","_"))

                # print('Question Entities:\n', match_kg, "\n") ### 與問題keywords相符的entities


                ### --- Entities Classification (需要時再跑就好)
                match_question_kg = []
                match_answer_kg = []
                ### Question kg
                for kg_entity in question_kg:
                    # print("Q's: ",kg_entity)
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
                    # print("A's: ",kg_entity)
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
                

                ### Search in Neo4j
                # Question
                all_question_keys = {'Disease': 0, 'Medical_Test': 0, 'Symptom': 0, 'Medication': 0}
                question_label_count = count_label_from_neo4j(match_question_kg)
                all_question_keys.update(question_label_count)
                # Percentages
                q_total = sum(all_question_keys.values())
                if q_total > 0:
                    percentages_q = {key: (value / q_total) * 100 for key, value in all_question_keys.items()}
                else:
                    percentages_q = {key: 0 for key in all_question_keys.keys()}
                print(percentages_q)


                # Answer
                all_ans_keys = {'Disease': 0, 'Medical_Test': 0, 'Symptom': 0, 'Medication': 0}
                ans_label_count = count_label_from_neo4j(match_answer_kg)
                all_ans_keys.update(ans_label_count)
                # Percentages
                a_total = sum(all_ans_keys.values())
                if a_total > 0:
                    percentages_a = {key: (value / a_total) * 100 for key, value in all_ans_keys.items()}
                else:
                    percentages_a = {key: 0 for key in all_ans_keys.keys()}
                print(percentages_a)


                ### write in csv
                if output_text:
                    with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/entities_classification/data_classification_final.csv', 'a+', newline='') as f6:
                        writer = csv.writer(f6)
                        writer.writerow([index+1, input_text[0], all_question_keys['Disease'], all_question_keys['Medical_Test'], 
                                        all_question_keys['Medication'], all_question_keys['Symptom'], 
                                        output_text[0], all_ans_keys['Disease'], all_ans_keys['Medical_Test'], 
                                        all_ans_keys['Medication'], all_ans_keys['Symptom']])
                        f6.flush()

                    with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/entities_classification/data_classification_percent_final.csv', 'a+', newline='') as f7:
                        writer = csv.writer(f7)
                        writer.writerow([index+1, input_text[0], percentages_q['Disease'], percentages_q['Medical_Test'], 
                                        percentages_q['Medication'], percentages_q['Symptom'], 
                                        output_text[0], percentages_a['Disease'], percentages_a['Medical_Test'], 
                                        percentages_a['Medication'], percentages_a['Symptom']])
                        f7.flush()
                else:
                    with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/entities_classification/data_classification_final.csv', 'a+', newline='') as f6:
                        writer = csv.writer(f6)
                        writer.writerow([index+1, input_text[0], all_question_keys['Disease'], all_question_keys['Medical_Test'], 
                                        all_question_keys['Medication'], all_question_keys['Symptom'], 
                                        "No answer output.",all_ans_keys['Disease'], all_ans_keys['Medical_Test'], 
                                        all_ans_keys['Medication'], all_ans_keys['Symptom']])
                        f6.flush()
                    
                    with open('/home/yuu0223/KG-Prompting-Based-on-Community-Search/output/CommunitySearch/entities_classification/data_classification_percent_final.csv', 'a+', newline='') as f7:
                        writer = csv.writer(f7)
                        writer.writerow([index+1, input_text[0], percentages_q['Disease'], percentages_q['Medical_Test'], 
                                        percentages_q['Medication'], percentages_q['Symptom'], 
                                        "No answer output.",percentages_a['Disease'], percentages_a['Medical_Test'], 
                                        percentages_a['Medication'], percentages_a['Symptom']])
                        f7.flush()
                ### ---
            
            except Exception as e:
                print(f"錯誤在第 {index+1} 題: {e}")
                continue
