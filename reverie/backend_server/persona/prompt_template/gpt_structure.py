"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modified for Local Ollama Inference (llama3.1:8b-instruct-q5_K_M)

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
import requests
import time 

from utils import *

# ============================================================================
# #####################[LOCAL OLLAMA CONFIGURATION] ##########################
# ============================================================================

# Chat requests go to Ollama's native API. Its OpenAI-compatible /v1 endpoint
# silently ignores the "options" field, so num_ctx, num_predict and
# repeat_penalty would never take effect there (the model runs at Ollama's
# default context and prompts past it are truncated without an error).
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# The OpenAI client is still used for embeddings.
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # Required by the client, ignored by Ollama
)

# Define the models to use
# LOCAL_CHAT_MODEL = "llama3.1:8b-instruct-q5_K_M"
LOCAL_CHAT_MODEL = "llama3.1:8b"
LOCAL_EMBEDDING_MODEL = "nomic-embed-text"

def temp_sleep(seconds=0.1):
    time.sleep(seconds)


# ============================================================================
# #####################[SECTION 1: CHAT STRUCTURE] ###########################
# ============================================================================

# Temperature to use on each attempt of a retry loop. At temperature 0 a retry
# sends the same prompt and gets the same answer back, so we loosen it a little
# on each attempt to give validation a real second chance.
RETRY_TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 0.9]

def retry_temperature(attempt, base=0.0):
    return max(base, RETRY_TEMPERATURES[min(attempt, len(RETRY_TEMPERATURES) - 1)])


# System message for prompts written for text-davinci completion: few-shot
# documents that stop mid-line and expect the model to just continue them.
# Without it, a chat model tends to re-answer every example and explain itself.
# Prompts opt in with "completion_mode": True in their gpt_param; open-ended
# prompts (e.g. wake up hour, daily plan) get worse with it, so it is not global.
COMPLETION_SYSTEM_PROMPT = (
    "You are a text completion engine. The user's message is a document that "
    "stops partway through. Reply with ONLY the text that continues it from "
    "exactly where it stops, following the format of the examples in it. Do "
    "not repeat the document or the examples, do not add commentary, and stop "
    "once the final item is complete."
)


def local_chat_request(prompt, temperature=0.0, num_predict=1024, system=None):
    """Single chat completion against the local Ollama model."""
    temp_sleep()
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": LOCAL_CHAT_MODEL,
                "messages": ([{"role": "system", "content": system}] if system else [])
                            + [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 8192,
                    "repeat_penalty": 1.1,     # Prevents the model from getting stuck in repetition loops
                    "num_predict": num_predict # Hard limit on output tokens (prevents infinite generation)
                }
            },
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        print(f"GPT_request ERROR: {type(e).__name__}: {e}")
        return "TOKEN LIMIT EXCEEDED"


def ChatGPT_single_request(prompt):
    return local_chat_request(prompt)


def GPT4_request(prompt, temperature=0.0):
    return local_chat_request(prompt, temperature)


def ChatGPT_request(prompt, temperature=0.0):
    return local_chat_request(prompt, temperature)


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
            curr_gpt_response = ChatGPT_request(prompt, retry_temperature(i)).strip()
            
            # Strip markdown code blocks if present
            curr_gpt_response = curr_gpt_response.replace("```json", "").replace("```", "").strip()
            
            # Try to extract JSON object
            start_index = curr_gpt_response.find('{')
            end_index = curr_gpt_response.rfind('}') + 1
            
            if start_index != -1 and end_index > start_index:
                curr_gpt_response = curr_gpt_response[start_index:end_index]
                parsed = json.loads(curr_gpt_response)
                if isinstance(parsed, dict) and "output" in parsed:
                    curr_gpt_response = parsed["output"]
                else:
                    curr_gpt_response = str(parsed)
            else:
                # No JSON found — use the raw response as the output
                # Strip common prefixes the LLM might add
                for prefix in ["Answer:", "answer:", "Output:", "output:", "Response:", "response:"]:
                    if curr_gpt_response.startswith(prefix):
                        curr_gpt_response = curr_gpt_response[len(prefix):].strip()
                # Remove surrounding quotes if present
                if curr_gpt_response.startswith('"') and curr_gpt_response.endswith('"'):
                    curr_gpt_response = curr_gpt_response[1:-1]
                if curr_gpt_response.startswith("'") and curr_gpt_response.endswith("'"):
                    curr_gpt_response = curr_gpt_response[1:-1]
            
            if func_validate(curr_gpt_response, prompt=prompt): 
                return func_clean_up(curr_gpt_response, prompt=prompt)
            
            if verbose: 
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")
        except Exception as e:
            if verbose: print(f"Safe Generate Error: {e}")
            pass

    # Return fail_safe instead of False to prevent None returns downstream
    return fail_safe_response


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
            curr_gpt_response = ChatGPT_request(prompt, retry_temperature(i)).strip()
            
            # Strip markdown code blocks if present
            curr_gpt_response = curr_gpt_response.replace("```json", "").replace("```", "").strip()
            
            # Try to extract JSON object
            start_index = curr_gpt_response.find('{')
            end_index = curr_gpt_response.rfind('}') + 1
            
            if start_index != -1 and end_index > start_index:
                curr_gpt_response = curr_gpt_response[start_index:end_index]
                parsed = json.loads(curr_gpt_response)
                if isinstance(parsed, dict) and "output" in parsed:
                    curr_gpt_response = parsed["output"]
                else:
                    curr_gpt_response = str(parsed)
            else:
                # No JSON found — use the raw response as the output
                # Strip common prefixes the LLM might add
                for prefix in ["Answer:", "answer:", "Output:", "output:", "Response:", "response:"]:
                    if curr_gpt_response.startswith(prefix):
                        curr_gpt_response = curr_gpt_response[len(prefix):].strip()
                # Remove surrounding quotes if present
                if curr_gpt_response.startswith('"') and curr_gpt_response.endswith('"'):
                    curr_gpt_response = curr_gpt_response[1:-1]
                if curr_gpt_response.startswith("'") and curr_gpt_response.endswith("'"):
                    curr_gpt_response = curr_gpt_response[1:-1]
            
            if func_validate(curr_gpt_response, prompt=prompt): 
                return func_clean_up(curr_gpt_response, prompt=prompt)
            
            if verbose: 
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")
        except Exception as e:
            if verbose: print(f"Safe Generate Error: {e}")
            pass

    # Return fail_safe instead of False to prevent None returns downstream
    return fail_safe_response


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

def GPT_request(prompt, gpt_parameter, temperature=None):
    # We route legacy completion calls through the chat endpoint for better
    # instruct compliance. The prompt's own "max_tokens" is not applied: those
    # were sized for text-davinci completions, and a chat model often restates
    # the few-shot examples before answering, so a tight cap cuts it off before
    # it reaches the answer.
    if temperature is None:
        temperature = gpt_parameter.get("temperature", 0.0)
    system = COMPLETION_SYSTEM_PROMPT if gpt_parameter.get("completion_mode") else None
    return local_chat_request(prompt, temperature, system=system)


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
        temperature = retry_temperature(i, gpt_parameter.get("temperature", 0.0))
        curr_gpt_response = GPT_request(prompt, gpt_parameter, temperature)
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
            input=[text],
            timeout=30.0
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding ERROR: {e}")
        # Return a dummy embedding of the correct dimension (nomic-embed-text is 768)
        # to prevent cascade crashes. It must be non-zero: cos_sim divides by the
        # vector norm, and a zero vector turns retrieval scores into NaN.
        return [1e-6] * 768


if __name__ == '__main__':
    # Simple test to verify the local setup works
    test_prompt = "You are a test agent. Output ONLY valid JSON: {'test': 'success'}"
    print("Testing ChatGPT_request...")
    print(ChatGPT_request(test_prompt))
    
    print("\nTesting get_embedding...")
    emb = get_embedding("hello world")
    print(f"Embedding generated successfully. Length: {len(emb)}")