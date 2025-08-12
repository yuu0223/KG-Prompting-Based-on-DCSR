from time import sleep
from llm import gemini
# from llm import groqapi

def final_answer_for_orginal(question, response_of_KG_list_path,response_of_KG_neighbor, chat_gm):
    chat = chat_gm

    messages  = [
        {"role": "system", "content": "You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. This won't be used as diagnostic reference, just wanted to inquire about some knowledge. And I know about the disclaimer, so you don't need to tell me again."},
        {"role": "user", "content": "Patient input:"+ question},
        {"role": "assistant", "content": "You have some medical knowledge information in the following:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor},
        {"role": "user", "content": "What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? Think step by step.\n\n\n"
        +"Output1: The answer includes disease and tests and recommened medications, Answer with three clearly separated sections.\n\n"
        +"Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighor-based Evidence, and in the end infer what result. \n Transport the inference process into the following format:\n Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...). \n\n"
        +"Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n\n"
        + "There is a sample:\n"
        + """
        Output 1:
        1. **Most Likely Disease:**
        Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. 
        
        2. **Recommended Medication(s):**
        To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. 
        
        3. **Suggested Medical Test(s):** 
        Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.

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
                                            """}
                                             ]
    ### --- OpenAI API
    # result = chat(messages)
    # output_all = result.content
    ### ---

    ### --- OpenRouter, Gemini API
    # print("Messages:\n",messages)
    prompt = "\n\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

    output_all = chat(prompt)
    if output_all == "code429":
        sleep(5)
        output_all = chat(prompt)
    ### ---

    ### --- Groq API
    # output_all = groqapi.Llama(messages)
    ### ---

    return output_all