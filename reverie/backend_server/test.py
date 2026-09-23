"""
File: test.py
Description: Test script to verify local LLM can generate multi-agent dialogue in JSON.
"""
import sys
import os
# Fix the import path issue
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import openai

# Initialize the modern OpenAI client pointing to your local Ollama server
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

LOCAL_CHAT_MODEL = "llama3.1:8b-instruct-q5_K_M"

def ChatGPT_request(prompt): 
    try: 
        response = client.chat.completions.create(
            model=LOCAL_CHAT_MODEL, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # CRITICAL: Forces the model to output ONLY valid JSON
            response_format={"type": "json_object"}, 
            extra_body={
                "options": {
                    "num_ctx": 8192
                }
            }
        )
        return response.choices[0].message.content
    
    except Exception as e: 
        print(f"Ollama Chat ERROR: {e}")
        return "Ollama ERROR"


prompt = """
---
Character 1: Maria Lopez is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
Character 2: Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.

Past Context: 
138 minutes ago, Maria Lopez and Klaus Mueller were already conversing about Maria's research paper mentioned by Klaus. This context takes place after that conversation.

Current Context: Maria Lopez was attending her Physics class (preparing for the next lecture) when Maria Lopez saw Klaus Mueller in the middle of working on his research paper at the library (writing the introduction).
Maria Lopez is thinking of initiating a conversation with Klaus Mueller.
Current Location: library in Oak Hill College

(This is what is in Maria Lopez's head: Maria Lopez should remember to follow up with Klaus Mueller about his thoughts on her research paper. Beyond this, Maria Lopez doesn't necessarily know anything more about Klaus Mueller) 

(This is what is in Klaus Mueller's head: Klaus Mueller should remember to ask Maria Lopez about her research paper, as she found it interesting that he mentioned it. Beyond this, Klaus Mueller doesn't necessarily know anything more about Maria Lopez) 

Here is their conversation. 

Maria Lopez: "
---
Output the response to the prompt above in json. The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"]. Output multiple utterances in the conversation until the conversation comes to a natural conclusion.
Example output json:
{"output": [["Jane Doe", "Hi!"], ["John Doe", "Hello there!"]]}
"""

print("Sending request to local Ollama (llama3.1:8b-instruct-q5_K_M)...")
print("-" * 50)

raw_response = ChatGPT_request(prompt)

# CRITICAL CLEANUP: Strip markdown backticks if the model adds them anyway
cleaned_response = raw_response.strip().replace("```json", "").replace("```", "").strip()

print(cleaned_response)
print("-" * 50)

# Try to parse it
try:
    parsed = json.loads(cleaned_response)
    print("\n✅ SUCCESS: Output is valid JSON!")
    print("Parsed output:", json.dumps(parsed, indent=2))
except json.JSONDecodeError as e:
    print(f"\n❌ WARNING: Output is NOT valid JSON. Error: {e}")