import bittensor as bt

from openai import OpenAI
import torch
from transformers import pipeline

import re
def clean_text(text):
    # Remove special characters with a regular expression
    text = re.sub(r'[^\w\s]', '', text)
    # Remove emojis and other unicode characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def get_llm_prompt(query: str) -> str:
    return f"Take the given query `{query}` and augment it to be more detailed using keywords. Do not make it longer than 30 words."
def get_llm_json_prompt(query: str) -> str:
    return f"Take the given query `{query}` and give me a list of 10 variations. Each variation should be augmented to have tangentially related items. For example, use synonyms, hyperbole, add specific names and types. Do not make any variation longer than 8 words. Return the list of variations in JSON"    
    
def get_llm_desc_prompt(query: str) -> str:
    return f"Take the given video description `{query}` and enhance it to be more detailed and interesting. Add richness but keep it relevant. Uniqueness is key. Do not use any special characters or emoticons. Keep the description to a 100-word limit."
def get_llm_write_desc_prompt(query: str) -> str:
    return f"Take the given video title `{query}` and write a detailed and interesting description about it. Add richness but keep it relevant. Uniqueness is key. Do not use any special characters or emoticons. Keep the description to a 100-word limit."

class AbstractAugment:
    def __init__(self, **kwargs):
        pass

    def __call__(self, query: str) -> str:
        try:
            new_query = self.augment_query(query)
            bt.logging.info(f"Augmented query: '{query}' -> '{new_query}'")
            return new_query
        except Exception as e:
            print(f"Error augmenting query: {e}")
            return query

    def augment_query(self, query: str) -> str:
        raise NotImplementedError


class NoAugment(AbstractAugment):
    def __init__(self, **kwargs):
        bt.logging.info("Running no query augmentation")

    def augment_query(self, query: str) -> str:
        return query


class LocalLLMAugment(AbstractAugment):
    def __init__(self, **kwargs):
        self.device = kwargs.get("device")
        if self.device == "cpu":
            raise ValueError("Cannot run Local LLM on CPU. Please move to a GPU instance or restart miner with `--neuron.query_augment OpenAIAugment` to use the GPT-4 API for augmenting instead of a local LLM.")
        model_name = "teknium/OpenHermes-2.5-Mistral-7B"
        self.pipe = pipeline("text-generation", model=model_name, device=self.device, torch_dtype=torch.float16, pad_token_id=32000)
        bt.logging.info(f"Running query augmentation with local LLM {model_name} (thanks Nous!)")

    def augment_query(self, query: str) -> str:
        prompt = f"""<|im_start|>system
        You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
        <|im_start|>user
        {get_llm_prompt(query)}<|im_end|>
        <|im_start|>assistant
        Detailed query: """
        new_query = self.pipe(prompt, max_new_tokens=64)[0]["generated_text"][len(prompt):].strip().strip("\"").strip("'")
        new_query = clean_text(new_query)
        return new_query
        
    def augment_description(self, query: str) -> str:
        prompt = f"""<|im_start|>system
        You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
        <|im_start|>user
        {get_llm_desc_prompt(query)}<|im_end|>
        <|im_start|>assistant
        Detailed query: """
        new_query = self.pipe(prompt, max_new_tokens=128)[0]["generated_text"][len(prompt):].strip().strip("\"").strip("'")
        return new_query
        
    def write_description(self, query: str) -> str:
        prompt = f"""<|im_start|>system
        You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
        <|im_start|>user
        {get_llm_write_desc_prompt(query)}<|im_end|>
        <|im_start|>assistant
        Detailed query: """
        new_query = self.pipe(prompt, max_new_tokens=128)[0]["generated_text"][len(prompt):].strip().strip("\"").strip("'")
        return new_query


class OpenAIAugment(AbstractAugment):
    def __init__(self, **kwargs):
        self.client = OpenAI()
        bt.logging.info("Running query augmentation with OpenAI GPT-4")

    def augment_query(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "user",
                    "content": get_llm_prompt(query)
                }
            ],
            temperature=0.5,
            max_tokens=64,
            top_p=1,
        )
        return response.choices[0].message.content.strip("\"").strip("'")
        
    def augment_json_query(self, query: str, temperature: float = 0.5) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "user",
                    "content": get_llm_json_prompt(query)
                }
            ],
            response_format= {
                "type": "json_object"
            },
            temperature=temperature,
            max_tokens=500,
            top_p=1,
        )
        print(response)
        return response.choices[0].message.content.strip("\"").strip("'")

    def augment_description(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "user",
                    "content": get_llm_desc_prompt(query)
                }
            ],
            temperature=0.1,
            max_tokens=64,
            top_p=1,
        )
        return response.choices[0].message.content.strip("\"").strip("'")

    def write_description(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "user",
                    "content": get_llm_write_desc_prompt(query)
                }
            ],
            temperature=0.1,
            max_tokens=64,
            top_p=1,
        )
        return response.choices[0].message.content.strip("\"").strip("'")
