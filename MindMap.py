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
import networkx as nx
from communitysearch import FindKG


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

    ### --- [Step 1]. Neo4j Sandbox Connection
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
    ### --- End of [Step 1]
    

    ### --- [Step 2]. Gemini API Connection
    GEMINI_API_KEY = os.getenv("gemini_api_key")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of [Step 2]


    ### --- [Step 3]. Extract Question & Answer's Entities
    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    # with open('./output/MindMap/Q714_Gemini15Flash/20250505_MM.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(['Question_ID', 'Question', 'MindMap'])


    with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open('./data/chatdoctor5k/keyword_embeddings_new.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)
    

    docs_dir = './data/chatdoctor5k/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)
    

    json_data = []
    SAVE_INTERVAL = 5
    with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
        lines = f.readlines()

    round_count = 0
    for line in tqdm(lines[round_count:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
        x = json.loads(line)
        print("Q_ID from data:", x["Q_ID"])
        input = x["qustion_output"]
        input = input.replace("\n","")
        input = input.replace("<OOS>","<EOS>")
        input = input.replace(":","") + "<END>"
        input_text = re.findall(re3,input)
        
        if input_text == []:
            continue
        print('Question:', x["Q_ID"],'\n',input_text[0])

        output_text = x["answer_output"]
        # output = x["answer_output"]
        # output = output.replace("\n","")
        # output = output.replace("<OOS>","<EOS>")
        # output = output.replace(":","") + "<END>"
        # output_text = re.findall(re3,output)
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

        # answer_kg = re.findall(re1,output)
        # if len(answer_kg) == 0:
        #     answer_kg = re.findall(re2,output)
        #     if len(answer_kg) == 0:
        #         print("<Warning> no entities found", output)
        #         continue
        # answer_kg = answer_kg[0].replace("<END>","").replace("<EOS>","")
        # answer_kg = answer_kg.replace("\n","")
        # answer_kg = answer_kg.split(", ")
        # print(answer_kg)

        
        match_kg = []
        entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
    

        for kg_entity in question_kg:
            
            keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
            max_index = cos_similarities.argmax()
                    
            match_kg_i = entity_embeddings["entities"][max_index]
            while match_kg_i.replace(" ","_") in match_kg:
                cos_similarities[max_index] = 0
                max_index = cos_similarities.argmax()
                match_kg_i = entity_embeddings["entities"][max_index]

            match_kg.append(match_kg_i.replace(" ","_"))

        print('Question Entities:\n', match_kg, "\n") ### 與問題keywords相符的entities
        ### --- End of [Step 3]

    
        # # 4. neo4j knowledge graph path finding
        if len(match_kg) != 1 or 0:
            start_entity = match_kg[0]
            candidate_entity = match_kg[1:]
            
            result_path_list = []
            while 1:
                flag = 0
                paths_list = []
                while candidate_entity != []:
                    end_entity = candidate_entity[0]
                    candidate_entity.remove(end_entity)                        
                    paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
                    path_list = []
                    if paths == [''] or paths == []:
                        flag = 1
                        if candidate_entity == []:
                            flag = 0
                            break
                        start_entity = candidate_entity[0]
                        candidate_entity.remove(start_entity)
                        break
                    else:
                        for p in paths:
                            path_list.append(p.split('->'))
                        if path_list != []:
                            paths_list.append(path_list)
                    
                    if exist_entity != {}:
                        try:
                            candidate_entity.remove(exist_entity)
                        except:
                            continue
                    start_entity = end_entity
                result_path = combine_lists(*paths_list)
            
            
                if result_path != []:
                    result_path_list.extend(result_path)                
                if flag == 1:
                    continue
                else:
                    break
                
            start_tmp = []
            for path_new in result_path_list:
            
                if path_new == []:
                    continue
                if path_new[0] not in start_tmp:
                    start_tmp.append(path_new[0])
            
            if len(start_tmp) == 0:
                    result_path = {}
                    single_path = {}
            else:
                if len(start_tmp) == 1:
                    result_path = result_path_list[:5]
                else:
                    result_path = []
                                            
                    if len(start_tmp) >= 5:
                        for path_new in result_path_list:
                            if path_new == []:
                                continue
                            if path_new[0] in start_tmp:
                                result_path.append(path_new)
                                start_tmp.remove(path_new[0])
                            if len(result_path) == 5:
                                break
                    else:
                        count = 5 // len(start_tmp)
                        remind = 5 % len(start_tmp)
                        count_tmp = 0
                        for path_new in result_path_list:
                            if len(result_path) < 5:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    if count_tmp < count:
                                        result_path.append(path_new)
                                        count_tmp += 1
                                    else:
                                        start_tmp.remove(path_new[0])
                                        count_tmp = 0
                                        if path_new[0] in start_tmp:
                                            result_path.append(path_new)
                                            count_tmp += 1

                                    if len(start_tmp) == 1:
                                        count = count + remind
                            else:
                                break

                try:
                    single_path = result_path_list[0]
                except:
                    single_path = result_path_list
                
        else:
            result_path = {}
            single_path = {} 
        
        

        # # 5. neo4j knowledge graph neighbor entities
        neighbor_list = []
        neighbor_list_disease = []
        for match_entity in match_kg:
            disease_flag = 0
            neighbors,disease = get_entity_neighbors(match_entity,disease_flag)
            neighbor_list.extend(neighbors)

            while disease != []:
                new_disease = []
                for disease_tmp in disease:
                    if disease_tmp in match_kg:
                        new_disease.append(disease_tmp)

                if len(new_disease) != 0:
                    for disease_entity in new_disease:
                        disease_flag = 1
                        neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                        neighbor_list_disease.extend(neighbors)
                else:
                    for disease_entity in disease:
                        disease_flag = 1
                        neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                        neighbor_list_disease.extend(neighbors)
        if len(neighbor_list)<=5:
            neighbor_list.extend(neighbor_list_disease)

        # print("neighbor_list: \n",neighbor_list)


        # 6. knowledge gragh path based prompt generation
        if len(match_kg) != 1 or 0:
            response_of_KG_list_path = []
            if result_path == {}:
                response_of_KG_list_path = []
            else:
                result_new_path = []
                for total_path_i in result_path:
                    path_input = "->".join(total_path_i)
                    result_new_path.append(path_input)
                
                # print("Pased-based path lists: \n", result_new_path)
                path = "\n".join(result_new_path)
                response_of_KG_list_path = prompt_path_finding(path) ###
        else:
            response_of_KG_list_path = '{}'

        # print("response_of_KG_list_path: \n", response_of_KG_list_path)
        # response_single_path = prompt_path_finding(single_path)

        
        # # 7. knowledge gragh neighbor entities based prompt generation   
        response_of_KG_list_neighbor = []
        neighbor_new_list = []
        for neighbor_i in neighbor_list:
            neighbor = "->".join(neighbor_i)
            neighbor_new_list.append(neighbor)

        if len(neighbor_new_list) > 5:
            neighbor_new_list = neighbor_new_list[:5]

        # print("Neighbor-based path lists: \n", neighbor_new_list)
        neighbor_input = "\n".join(neighbor_new_list)


        response_of_KG_neighbor = prompt_neighbor(neighbor_input) ###
        # print("neighbor_input: \n", neighbor_input) ###


        print("Start to generate final answer...")
        # # 8. prompt-based medical diaglogue answer generation
        # 若沒有回答出來，讓 LLM 重新回答
        times=1
        while True:
            output_all = FinalPrompt.final_answer_for_orginal(input_text[0],response_of_KG_list_path,response_of_KG_neighbor, chat_gm)
            # print("output_all: \n", output_all)
            output1 = extract_final_answer(output_all)
            # output1, output2, output3 = extract_final_answer(output_all)
            if len(output1) > 0:
                output1 = output1[0]
                # output2 = output2[0]
                # output3 = output3[0]
                break
            elif times == 2:
                print("second round.")
                break
            else:
                times+=1

        # print("final_ans: ", output1)

        
        # # ### save the final result
        ### --- save for json
        json_data.append({
            "Q_ID": x["Q_ID"],
            "question": input_text[0],
            "query_nodes": match_kg,
            "path-based": result_new_path,
            "neighbor-based": neighbor_new_list,
            "output": output_all
        })

        # # 每 SAVE_INTERVAL 筆就存檔一次
        # # if (round_count + 1) % SAVE_INTERVAL == 0:
        # with open("./output/MindMap/Final_Q714/20250408_MM_Gemini20Flash.json", "w", encoding="utf-8") as f:
        #     json.dump(json_data, f, ensure_ascii=False, indent=4)
        ### ---

        with open('./output/MindMap/Q714_Gemini15Flash/20250505_MM.csv', 'a+', newline='') as f6:
            writer = csv.writer(f6)
            writer.writerow([x["Q_ID"], input_text[0], output1])
            f6.flush()
            
    driver.close()
               