from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import os, copy, types, gc, sys
from pathlib import Path

# Chatbot server configuration
HOST = 'localhost' 
PORT = 8001
URL = f'http://{HOST}:{PORT}'

# RWKV model configuration
STRATEGY = 'cuda fp16i8'
MODEL_NAME = 'RWKV-4-Pile-14B-20230313-ctx8192-test1050.pth'
N_LAYER = 32
N_EMBD = 4096
CTX_LEN = 8192 

# Prompt configuration
CHAT_LANG = "English"  
PROMPT_FILE = f'{Path(__file__).parent}/init_prompt/English-1.py'

# Assign visible GPU devices 
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True  
torch.backends.cuda.matmul.allow_tf32 = True

# RWKV environment variables
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

# Model file path  
MODEL_PATH = f'{Path(__file__).parent}/data/{MODEL_NAME}'

# Generation settings
CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150  
FREE_GEN_LEN = 200
GEN_TEMP = 1.0
GEN_TOP_P = 0.8
GEN_ALPHA_PRES = 0.2
GEN_ALPHA_FREQ = 0.2

CHUNK_LEN = 256

########################################################################################################

chat_states = {}

print(f'\n{CHAT_LANG} - {STRATEGY} - {PROMPT_FILE}')

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

# Load the prompt 
with open(PROMPT_FILE, 'rb') as file:
    user = None
    bot = None
    interface = None  
    init_prompt = None
    exec(compile(file.read(), PROMPT_FILE, 'exec'))

init_prompt = '\n' + '\n'.join(init_prompt.strip().split('\n')).strip() + '\n\n'

# Load the model
print(f'Loading model - {MODEL_PATH}')
model = RWKV(model=MODEL_PATH, strategy=STRATEGY)  

pipeline = PIPELINE(model, f"{Path(__file__).parent}/20B_tokenizer.json")
END_OF_TEXT = 0
END_OF_LINE = 187

model_tokens = []
model_state = None
    
def run_rnn(token_ids, newline_adj=0):
    global model_tokens, model_state
    
    model_tokens += token_ids
    
    while len(token_ids) > 0:
        out, model_state = model.forward(token_ids[:CHUNK_LEN], model_state)
        token_ids = token_ids[CHUNK_LEN:]
        
    out[END_OF_LINE] += newline_adj
    return out

def save_state(index, key, out):  
    chat_states[f'{key}_{index}'] = {
        'out': out,
        'rnn': copy.deepcopy(model_state),
        'tokens': copy.deepcopy(model_tokens)
    }
    
def load_state(index, key):
    global model_tokens, model_state
    
    state = chat_states[f'{key}_{index}']
    model_state = copy.deepcopy(state['rnn']) 
    model_tokens = copy.deepcopy(state['tokens'])
    
    return state['out']

########################################################################################################

out = run_rnn(pipeline.encode(init_prompt))
save_state('', 'chat_init', out)

gc.collect()  
torch.cuda.empty_cache()

for idx in ['dummy_server']:
    save_state(idx, 'chat', out)
    
def handle_message(msg):
    global model_tokens, model_state
    
    msg = msg.replace('\\n','\n').strip()
    
    temp = GEN_TEMP
    top_p = GEN_TOP_P
    
    if "-temp=" in msg:
        temp = float(msg.split("-temp=")[1].split(" ")[0]) 
        msg = msg.replace(f"-temp={temp:g}","")
    if "-top_p=" in msg:    
        top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace(f"-top_p={top_p:g}","")
        
    temp = max(0.2, min(5.0, temp))
    top_p = max(0.0, top_p)
        
    if msg == '+reset':
        out = load_state('', 'chat_init')  
        save_state('dummy_server', 'chat', out)
        return "Chat reset done."
    
    elif msg.startswith('+gen ') or msg.startswith('+i ') or msg.startswith('+qa ') or msg.startswith('+qq ') or msg == '+++' or msg == '++':
        
        if msg.startswith('+gen '):
            new_prompt = '\n' + msg[5:].strip()
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new_prompt))  
            save_state('dummy_server', 'gen_0', out)
            
        elif msg.startswith('+i '):
            new_prompt = f'''\nBelow is a task instruction. Write an appropriate response to complete the request.\n\n# Instruction:\n{msg[3:].strip()}\n\n# Response:\n'''
            model_state = None 
            model_tokens = []
            out = run_rnn(pipeline.encode(new_prompt))
            save_state('dummy_server', 'gen_0', out)
            
        elif msg.startswith('+qq '):
            new_prompt = '\nQ: ' + msg[4:].strip() + '\nA:'  
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new_prompt))
            save_state('dummy_server', 'gen_0', out)
            
        elif msg.startswith('+qa '):  
            out = load_state('', 'chat_init')

            new_prompt = f"{user}{interface} {msg[4:].strip()}\n\n{bot}{interface}"
            
            out = run_rnn(pipeline.encode(new_prompt))
            save_state('dummy_server', 'gen_0', out)
            
        elif msg == '+++':
            try:
                out = load_state('dummy_server', 'gen_1')
                save_state('dummy_server', 'gen_0', out) 
            except:
                pass
            
        elif msg == '++':  
            try:
                out = load_state('dummy_server', 'gen_0')
            except:
                pass
            
        begin = len(model_tokens)  
        out_last = begin
        occurrence = {}
        for i in range(FREE_GEN_LEN + 100):
            for token_id, count in occurrence.items():
                out[token_id] -= (GEN_ALPHA_PRES + count * GEN_ALPHA_FREQ) 
                
            token = pipeline.sample_logits(out, temperature=temp, top_p=top_p)
                
            if token == END_OF_TEXT:
                break
            
            occurrence[token] = occurrence.get(token, 0) + 1

            if msg.startswith('+qa '):  
                out = run_rnn([token], newline_adj=-2)
            else:
                out = run_rnn([token])
                
            decoded = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in decoded:
                print(decoded, end='', flush=True) 
                out_last = begin + i + 1
                if i >= FREE_GEN_LEN:
                    break
                  
        print('\n')
        response = pipeline.decode(model_tokens[begin:]).strip()
        save_state('dummy_server', 'gen_1', out)
        
        return response
        
    else:
        
        if msg == '+':
            try:
                out = load_state('dummy_server', 'chat_pre')
            except:
                pass  
        else:
            out = load_state('dummy_server', 'chat') 
            new_prompt = f"{user}{interface} {msg}\n\n{bot}{interface}"
            out = run_rnn(pipeline.encode(new_prompt), newline_adj=-999999999)
            save_state('dummy_server', 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        
        occurrence = {}
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= CHAT_LEN_SHORT:
                newline_adj = (i - CHAT_LEN_SHORT) / 10
            elif i <= CHAT_LEN_LONG:
                newline_adj = 0  
            else:
                newline_adj = min(2, (i - CHAT_LEN_LONG) * 0.25)

            for token, count in occurrence.items():
                out[token] -= (GEN_ALPHA_PRES + count * GEN_ALPHA_FREQ)
                
            token = pipeline.sample_logits(out, temperature=temp, top_p=top_p)
            
            occurrence[token] = occurrence.get(token, 0) + 1
            
            out = run_rnn([token], newline_adj=newline_adj)
            out[END_OF_TEXT] = -999999999

            decoded = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in decoded:
                out_last = begin + i + 1

            response = pipeline.decode(model_tokens[begin:]) 
            if '\n\n' in response:
                response = response.strip()
                break
            
        save_state('dummy_server', 'chat', out)
            
        return response
        
app = FastAPI()

@app.get("/chat_api")
async def chat(text:str=""):
    reply = handle_message(text).replace('\n','<br>')
    print(f'Input: {text}\nReply: {reply}')
    
    return {
        "output": [
            {
                "type": "text",
                "value": reply
            }
        ]
    }

app.mount("/", StaticFiles(directory="html", html=True), name="html")

def start_server():
    uvicorn.run(app, host=HOST, port=PORT)
    
if __name__ == '__main__':
    start_server()
