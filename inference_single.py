import os
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import BitsAndBytesConfig, AutoTokenizer, GPTNeoXForCausalLM
import json
import argparse
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument('--sentence_1',type=str,help='you should writing sentence_1')
p.add_argument('--sentence_3',type=str,help='you should writing sentence_3')
config = p.parse_args()

Qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules="embed_in,embed_out",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
)


tokenizer = AutoTokenizer.from_pretrained('EleutherAI/polyglot-ko-12.8b')
model=GPTNeoXForCausalLM.from_pretrained('EleutherAI/polyglot-ko-12.8b', quantization_config=Qconfig)

model_id_='output/sft_lora_model'
model=PeftModel.from_pretrained(model,model_id_)

PROMPT_TEMPLATE = (
    "[INST] <<SYS>>\n"
    "You are a helpful AI assistant. 당신은 유능한 AI 어시스턴트 입니다. 당신은 일관성과 논리적 흐름을 유지하면서 문장1과 문장3을 자연스럽게 연결하는 역할을 하는 문장 2를 생성합니다.\n"
    "<</SYS>>\n\n{instruction} [/INST]"
)


sentence1 = '문장1: '+config.sentence_1
sentence3 = '문장3: '+config.sentence_3
input_text = f"{sentence1}\n{sentence3}\n"
source = PROMPT_TEMPLATE.format_map({'instruction':input_text})
tokened = tokenizer(source, return_tensors="pt").to(device)
with torch.no_grad():
    bogan_ids = model.generate(input_ids=tokened.input_ids, attention_mask=tokened.attention_mask ,max_length=512,
        do_sample=True,
        top_k=20,
        top_p=0.92,
        num_beams=5,
        eos_token_id=tokenizer.eos_token_id
    )

bogan = tokenizer.decode(bogan_ids[0], skip_special_tokens=True)

print(bogan[143:])
        
        