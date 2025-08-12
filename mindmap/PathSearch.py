import MindMap
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    )
from llm import gemini
import os
from time import sleep

GEMINI_API_KEY = os.getenv("gemini_api_key")
chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)

def find_shortest_path(start_entity_name, end_entity_name,candidate_list, driver):
    global exist_entity
    exist_entity = {}
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


def prompt_path_finding(path_input):
    chat = chat_gm

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
    # print(type(chat_prompt_with_values.to_messages()[0]))

    ### --- OpenAI API
    # response_of_KG_path = MindMap.call_openai_api(chat_prompt_with_values.to_messages()[0]).content
    # print("check1:", chat_prompt_with_values.to_messages()[0].content)
    ### ---

    ### --- Gemini
    response_of_KG_path = chat(chat_prompt_with_values.to_messages()[0].content)
    if response_of_KG_path == "code429":
        sleep(10)
        response_of_KG_path = chat(chat_prompt_with_values.to_messages()[0].content)
    # print("check:",response_of_KG_path )
    ### ---

    return response_of_KG_path


## Step 4: neo4j knowledge graph path finding
def PathFinding(match_kg, driver):
    if len(match_kg) != 1 or 0:
        start_entity = match_kg[0]
        candidate_entity = match_kg[1:]
        
        result_path_list = []
        while 1:
            flag = 0
            paths_list = []
            ### 尋找每個keywords的shortest path，以5hop為限。
            while candidate_entity != []:
                end_entity = candidate_entity[0]
                candidate_entity.remove(end_entity)                      
                paths, exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity, driver)
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
            result_path = MindMap.combine_lists(*paths_list)
        
        
            if result_path != []:
                result_path_list.extend(result_path)                
            if flag == 1:
                continue
            else:
                break
        
        # print("result_path_list: \n", result_path_list)
        # [print(i) for i in result_path_list]
        # print("result_path_list_len: \n", len(result_path_list))
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
            # 如果找到的所有路徑start entity只有1個，就直接取前5條路徑
            if len(start_tmp) == 1:
                result_path = result_path_list[:5]
            else:
                result_path = []
                                            
                if len(start_tmp) >= 5:
                    for path_new in result_path_list:
                        if path_new == []:
                            continue
                        # 如果start entity超過5個的話，從前幾條path中找出5個不同的start entity的路徑
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
    
    print("result_path: ", result_path)
    print("single_path: ", single_path)
    
    return result_path, single_path



# Step 6: knowledge gragh path based prompt generation
def PathBasedPromptGenerate(match_kg, result_path, single_path, evidence_node):
    print('check: \n', result_path)
    if len(match_kg) != 1 or 0:
        response_of_KG_list_path = ''
        if result_path == {}:
            response_of_KG_list_path = ''
        else:
            result_new_path = []
            for total_path_i in result_path:
                path_input = "->".join(total_path_i)
                result_new_path.append(path_input)
                #### ---- 計算evidence節點數用
                [evidence_node.add(entity) for entity in total_path_i if entity.istitle()]
                print("evidence node: ", evidence_node)
                ############################


    ### 這部分計算節點數時先不跑
            # path = "\n".join(result_new_path)
            # response_of_KG_list_path = prompt_path_finding(path)
            # # if MindMap.is_unable_to_answer(response_of_KG_list_path, chat_gm):
            # #     response_of_KG_list_path = prompt_path_finding(path)
            # # print("response_of_KG_list_path",response_of_KG_list_path)
    ###
    else:
        response_of_KG_list_path = '{}'

    ### response_single_path = "temp"
    # response_single_path = prompt_path_finding(single_path)
    # if MindMap.is_unable_to_answer(response_single_path, chat_gm):
    #     response_single_path = prompt_path_finding(single_path)
    
    # return response_of_KG_list_path, response_single_path, evidence_node
    return response_of_KG_list_path, evidence_node