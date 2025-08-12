import requests
import json
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

class Gemma3(LLM):
    API_KEY: str

    @property
    def _llm_type(self) -> str:
        return "Gemma3-4B"

    def _call(
        self,
        prompt: Any,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.API_KEY}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "google/gemma-3-4b-it:free", # Optional
                # {"role": "user", "content": [{"type": "text", "text": prompt}]}
                "messages": prompt
            })
        )
        
        try:
            print("json:\n", response.json())
            output = response.json()["choices"][0]["message"]["content"]
        except KeyError:
            #目前先設定超過額度或速率，之後再來改
            output = "code429"

        return output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"API_KEY": self.API_KEY}