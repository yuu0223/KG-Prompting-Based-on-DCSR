import MindMap
from typing import Dict, List
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    )
from time import sleep
from llm import gemini
import os

GEMINI_API_KEY = os.getenv("gemini_api_key")
chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)


def get_entity_neighbors(entity_name: str,disease_flag, session) -> List[List[str]]:
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


def prompt_neighbor(neighbor):
    chat = chat_gm

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
    # print("Neighbor: \n",chat_prompt_with_values.to_messages()[0].content)

    ### --- OpenAI API
    # response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content
    ### ---

    ### --- Gemini
    response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()[0].content)
    if response_of_KG_neighbor == "code429":
        sleep(5)
        response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()[0].content)
    ### ---

    return response_of_KG_neighbor



## Step 5: neo4j knowledge graph neighbor entities
def NeighborFinding(match_kg, driver):
    session = driver.session()
    neighbor_list = []
    neighbor_list_disease = []
    for match_entity in match_kg:
        disease_flag = 0
        neighbors,disease = get_entity_neighbors(match_entity, disease_flag, session)
        neighbor_list.extend(neighbors)

        while disease != []:
            new_disease = []
            for disease_tmp in disease:
                if disease_tmp in match_kg:
                    new_disease.append(disease_tmp)

            if len(new_disease) != 0:
                for disease_entity in new_disease:
                    disease_flag = 1
                    neighbors,disease = get_entity_neighbors(disease_entity,disease_flag, session)
                    neighbor_list_disease.extend(neighbors)
            else:
                for disease_entity in disease:
                    disease_flag = 1
                    neighbors,disease = get_entity_neighbors(disease_entity,disease_flag, session)
                    neighbor_list_disease.extend(neighbors)
    if len(neighbor_list)<=5:
        neighbor_list.extend(neighbor_list_disease)

    print("final_neighbor_list",neighbor_list)
    return neighbor_list

#Step 7: knowledge gragh neighbor entities based prompt generation
def NeighborBasedPromptGenerate(neighbor_list, evidence_node):
    response_of_KG_list_neighbor = []
    neighbor_new_list = []
    for neighbor_i in neighbor_list:
        neighbor = "->".join(neighbor_i)
        neighbor_new_list.append(neighbor)

    if len(neighbor_new_list) > 5:
        neighbor_new_list_used = neighbor_new_list[:5]
        neighbor_input = "\n".join(neighbor_new_list_used)
        neighbor_paths_num = 5
    else:
        neighbor_new_list_used = neighbor_new_list[:len(neighbor_new_list)]
        neighbor_input = "\n".join(neighbor_new_list_used)
        neighbor_paths_num = len(neighbor_new_list)
    # print("neighbor_input:\n", neighbor_input)
    
    ### ---- 計算Neighbor節點數用
    neighbor_node = list()
    for node_list in neighbor_new_list_used:
        node = node_list.split("->")[2]
        [neighbor_node.append(nodes) for nodes in node.split(",")]
    [evidence_node.add(nodes) for nodes in neighbor_node]
    ############################
    
    ### 這部分計算節點數時先不跑
    # ### response_of_KG_neighbor = "temp"
    # response_of_KG_neighbor = prompt_neighbor(neighbor_input)
    # # if MindMap.is_unable_to_answer(response_of_KG_neighbor, chat_gm):
    # #     response_of_KG_neighbor = prompt_neighbor(neighbor_input)
    # # print("response_of_KG_neighbor",response_of_KG_neighbor)

    # return response_of_KG_neighbor, evidence_node, neighbor_paths_num
    return evidence_node, neighbor_paths_num