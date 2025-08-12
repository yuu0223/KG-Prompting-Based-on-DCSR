### 在嘗試class會有import迴圈的問題，待解決，目前只能用def

import os
import groq
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

def Llama(prompt, model_name='llama-3.1-70b-versatile'):

    client = Groq(api_key=API_KEY)

    try:
        chat_completion = client.chat.completions.create(
            # messages=[
            #     {
            #         "role": "system",
            #         "content": "You are a helpful assistant.",
            #     },
            #     {
            #         "role": "user",
            #         "content": "Explain the importance of low latency LLMs",
            #     },
            # ],
            messages=prompt,
            model=model_name,
        )
        
        response = chat_completion.choices[0].message.content
        return response

    except groq.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except groq.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except groq.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)
        # print(e.body["error"]["code"]) #出錯原因



# import os
# from dotenv import load_dotenv
# load_dotenv()
# import groq
# from groq import Groq
# API_KEY = os.getenv("GROQ_API_KEY")

# from langchain_core.language_models.llms import LLM
# from typing import Any, List, Mapping, Optional
# from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# import warnings
# warnings.filterwarnings("ignore")


# class SystemMessage:
#     def __init__(self, content: str):
#         self.role = "system"
#         self.content = content

# class HumanMessage:
#     def __init__(self, content: str):
#         self.role = "user"
#         self.content = content

# class AIMessage:
#     def __init__(self, content: str):
#         self.role = "assistant"
#         self.content = content


# class Llama(LLM):
#     API_KEY: str

#     @property
#     def _llm_type(self) -> str:
#         return "Llama3.1"

#     def _call(
#         self,
#         prompt: List[Any],
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#         ) -> List[Any]:

#         client = Groq(api_key=self.API_KEY)

#         formatted_messages = [{"role": pt.role, "content": pt.content} for pt in prompt]

#         try:
#             chat_completion = client.chat.completions.create(
#                 messages=formatted_messages,
#                 model="llama-3.1-70b-reasoning",
#             )
            
#             response = chat_completion.choices[0].message.content
#             return response
        
#         except groq.APIConnectionError as e:
#             print("The server could not be reached")
#             print(e.__cause__)  # an underlying Exception, likely raised within httpx.
#         except groq.RateLimitError as e:
#             print("A 429 status code was received; we should back off a bit.")
#         except groq.APIStatusError as e:
#             print("Another non-200-range status code was received")
#             print(e.status_code)
#             # print(e.response)
#             print(e.body["error"]) #出錯原因
    
#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         """Get the identifying parameters."""
#         return {"API_KEY": self.API_KEY}
    

# prompt =[{"role": "system", "content": "You are a helpful assistant."},
# {"role": "user", "content":" How's the weather now in America？"}]
# print(Llama(prompt))
