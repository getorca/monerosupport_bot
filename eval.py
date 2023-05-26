import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
import praw
import os


# generation configs
temperature = 0.7
top_p=0.1
top_k=40
typical_p=1
repition_penalty=1.18
encoder_repetion_penalty=1
no_repeat_ngram_size=0
length_penalty=1
num_beams=1
max_new_tokens=512
do_sample=True
# reddit configs
r_client_id=os.getenv('R_CLIENT_ID')
r_client_secret=os.getenv('R_CLIENT_SECRET')
r_user_agent=os.getenv('R_USER_AGENT')
# model configs
lora_tune = '/home/lawrence/monerosupport_bot/data/lora_1'
base_model = '/files/my_llama_hf/llama_hf_7B'
device = "cuda"


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map = {"": 0}  # force to use 1 GPU
)

model = PeftModel.from_pretrained(
    model,
    lora_tune,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model)

generation_config = GenerationConfig(
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    num_beams=num_beams,
    do_sample=do_sample,
    repition_penalty=repition_penalty,
    encoder_repetion_penalty=encoder_repetion_penalty
    
)

def extract_response(output):
    '''
    extracts the text first  response
    '''
    start = output.find('<|RESPONSE|>')+12
    end = output.find('<|END_RESPONSE|>')
    first_response = output[start:end]
    return first_response

def generate(input_ids):
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            # return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
        )
    
    return tokenizer.decode(generation_output[0])

def create_tokenized_prompt(title, post):
    prompt = '''
    <|SYSTEM|>
    You are a help monero support assitant on reddit. You are tasked with responding to user support questions with helpful and accurate knowledge of monero.insightful and helpful replies. User inputs are sent as JSON, you will respond with markdown on reddit.
    <|END_SYSTEM|>
    <|USER_INPUT|>
    {
    "title": "%s", 
    "input": "%s"
    }
    <|END_USER_INPUT|>
    <|RESPONSE|> 
    ''' % (title, input)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    return input_ids

def get_response(title, post):
    tokenized_prompt = create_tokenized_prompt(title, post)
    raw_response = generate(tokenized_prompt)
    response = extract_response(raw_response)
    return response



if __name__ ==  '__main__':
    
    reddit = praw.Reddit(
        client_id='CXgNDZ2PlJHqgAEbjpHIqA',
        client_secret='K7OYX62-RtG_Wq-RDZqpsZn77-QO6Q',
        user_agent='YFNAR'
    )
    
    subreddit = reddit.subreddit("monerosupport")

    for submission in subreddit.new(limit=100):
        post = {
            'title': submission.title,
            'selftext': submission.selftext,
            'link': submission.permalink,
            'id': submission.id,
            'ai_response': get_response(submission.title, submission.selftext)
        }
        with open('data/eval_1.jsonl', 'a') as f:
            f.write(json.dumps(post) + '\n')