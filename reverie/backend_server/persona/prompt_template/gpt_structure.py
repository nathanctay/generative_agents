"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modified for Local Ollama Inference (Qwen 2.5 7B)

File: gpt_structure.py
Description: Wrapper functions for calling local Ollama APIs via OpenAI-compatible client.
"""

import sys
import os
# Add the backend_server directory to the path so 'utils' can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import openai
import time 

from utils import *

# ============================================================================
# #####################[LOCAL OLLAMA CONFIGURATION] ##########################
# ============================================================================

# Initialize the modern OpenAI client pointing to your local Ollama server
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # Required by the client, ignored by Ollama
)

# Define the models to use
LOCAL_CHAT_MODEL = "qwen2.5:7b"
LOCAL_EMBEDDING_MODEL = "nomic-embed-text"

def temp_sleep(seconds=0.1):
    time.sleep(seconds)


# ============================================================================
# #####################[SECTION 1: CHAT STRUCTURE] ###########################
# ============================================================================

def ChatGPT_single_request(prompt): 
    temp_sleep()
    try:
        response = client.chat.completions.create(
            model=LOCAL_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # Keep deterministic for simulation logic
            extra_body={
                "options": {
                    "num_ctx": 8192 # CRITICAL: Prevents memory truncation!
                }
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ollama Chat ERROR: {e}")
        return "Ollama ERROR"


def GPT4_request(prompt): 
    temp_sleep()
    try: 
        # Route GPT-4 calls to our local model as well
        response = client.chat.completions.create(
            model=LOCAL_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body={"options": {"num_ctx": 8192}}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ollama Chat ERROR: {e}")
        return "Ollama ERROR"


def ChatGPT_request(prompt): 
    try: 
        response = client.chat.completions.create(
            model=LOCAL_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body={"options": {"num_ctx": 8192}}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ollama Chat ERROR: {e}")
        return "Ollama ERROR"


def GPT4_safe_generate_response(prompt, example_output, special_instruction, repeat=3, fail_safe_response="error", func_validate=None, func_clean_up=None, verbose=False): 
    prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose: 
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat): 
        try: 
            curr_gpt_response = GPT4_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]
            
            if func_validate(curr_gpt_response, prompt=prompt): 
                return func_clean_up(curr_gpt_response, prompt=prompt)
            
            if verbose: 
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")
        except Exception as e:
            if verbose: print(f"Safe Generate Error: {e}")
            pass

    return False


def ChatGPT_safe_generate_response(prompt, example_output, special_instruction, repeat=3, fail_safe_response="error", func_validate=None, func_clean_up=None, verbose=False): 
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose: 
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat): 
        try: 
            curr_gpt_response = ChatGPT_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]
            
            if func_validate(curr_gpt_response, prompt=prompt): 
                return func_clean_up(curr_gpt_response, prompt=prompt)
            
            if verbose: 
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")
        except Exception as e:
            if verbose: print(f"Safe Generate Error: {e}")
            pass

    return False


def ChatGPT_safe_generate_response_OLD(prompt, repeat=3, fail_safe_response="error", func_validate=None, func_clean_up=None, verbose=False): 
    if verbose: 
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat): 
        try: 
            curr_gpt_response = ChatGPT_request(prompt).strip()
            if func_validate(curr_gpt_response, prompt=prompt): 
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose: 
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")
        except: 
            pass
    print("FAIL SAFE TRIGGERED") 
    return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL COMPLETION STRUCTURE] ##############
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
    temp_sleep()
    try: 
        # We route legacy completion calls through the chat endpoint for better instruct compliance
        response = client.chat.completions.create(
            model=LOCAL_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=gpt_parameter.get("temperature", 0.0),
            extra_body={"options": {"num_ctx": 8192}}
        )
        return response.choices[0].message.content
    except Exception as e: 
        print(f"TOKEN LIMIT EXCEEDED or Ollama ERROR: {e}")
        return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
    if type(curr_input) == type("string"): 
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):   
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt: 
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(prompt, gpt_parameter, repeat=5, fail_safe_response="error", func_validate=None, func_clean_up=None, verbose=False): 
    if verbose: 
        print(prompt)

    for i in range(repeat): 
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        if func_validate(curr_gpt_response, prompt=prompt): 
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose: 
            print("---- repeat count: ", i, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")
    return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if not text: 
        text = "this is blank"
    try:
        # Use local Ollama embedding model
        response = client.embeddings.create(
            model=LOCAL_EMBEDDING_MODEL, 
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding ERROR: {e}")
        # Return a dummy embedding of the correct dimension (nomic-embed-text is 768) to prevent cascade crashes
        return [0.0] * 768


if __name__ == '__main__':
    # Simple test to verify the local setup works
    test_prompt = "You are a test agent. Output ONLY valid JSON: {'test': 'success'}"
    print("Testing ChatGPT_request...")
    print(ChatGPT_request(test_prompt))
    
    print("\nTesting get_embedding...")
    emb = get_embedding("hello world")
    print(f"Embedding generated successfully. Length: {len(emb)}")