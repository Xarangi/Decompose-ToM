import os
import openai
from openai import OpenAI
import time
from typing import *
import google.genai as genai
from google.genai.types import HarmCategory, HarmBlockThreshold

class LanguageModel:
    def __init__(self, model_name, api_key=None, temperature=0.0, model_type="openai"):
        self.model_name = model_name
        self.model_type = model_type
        if self.model_type not in ["openai", "gemini", "local"]:
            print("Error: Model type can only be openai, gemini, local. Using openai.")
        self.temperature = temperature
        self.model=None
        if api_key == None:
            if self.model_type=="openai":
                api_key = os.getenv("OPENAI_API_KEY") # Setting OpenAI API key if provided
            elif self.model_type=="gemini":
                api_key = os.getenv("GEMINI_API_KEY")
            else:
                api_key = "token123"
        if self.model_type=="openai":
            openai_client = OpenAI(api_key=api_key)
            self.model = openai_client
        elif self.model_type=="gemini":
            genai.configure(api_key=api_key)
            generation_config = genai.GenerationConfig(temperature=0)
            self.model = genai.GenerativeModel(model_name = self.model_name, generation_config = generation_config)
        else:
            openai_client = OpenAI(
                base_url="http://localhost:30000/v1",
                api_key=api_key,
            )
            self.model = openai_client
    def get_output(self, prompt:str, retry_count=10) -> str:
        if self.model_type =="openai" or self.model_type=="local":
            messages = [
                {"role": "system", "content": f"{prompt}"},
            ]

            while retry_count > 0:
                try:
                    res = self.model.chat.completions.create(
                        model=self.model_name,
                        # reasoning_effort="high",
                        messages=messages,
                        temperature = self.temperature
                    )
                    output = res.choices[0].message.content
                    
                    return output
                except ValueError as e:
                    print(f"Attempt failed with error: {e}")
                    retry_count -= 1
                    if retry_count > 0:
                        print("Retrying...")
                        time.sleep(10)
                    else:
                        print("All attempts failed.")

        if self.model_type == "gemini":
            while retry_count > 0:
                try:
                    ans = model.generate_content(prompt, safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    })
                    return ans.text.strip().strip(".")
                except Exception as e:
                    print(f"Attempt failed with error: {e}")
                    retry_count -= 1
                    if retry_count > 0:
                        print("Retrying...")
                        time.sleep(10)
                    else:
                        print("All attempts failed.")

