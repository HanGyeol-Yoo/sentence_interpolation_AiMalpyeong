import os
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import BitsAndBytesConfig, AutoTokenizer, GPTNeoXForCausalLM
import json
import argparse
from peft import PeftModel

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

json_file_path = 'data/test/test.json'

with open(json_file_path,'r',encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    
bogan_list=[]

for i in range(len(json_data)):
    sentence1 = json_data[i]["input"].split("\n")[0].replace("", "")
    sentence2 = json_data[i]["input"].split("\n")[1].replace("", "")

    input_text = f"{sentence1}\n{sentence2}\n"
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

        bogan_list.append(bogan[143:])
    print(f"{i+1}번째 완료")
        
ready_list=[]
for text in bogan_list:
    index = text.find('문장2:')
    if index != -1:
        sentence2 = text[index+5:]
        ready_list.append(sentence2)


file_path = "origindata/nikluge-sc-2023-test.jsonl"
output_file_path = "outputs.jsonl"

with open(file_path, "r", encoding="utf-8") as input_file:

    lines = input_file.readlines()

    for i, line in enumerate(lines):
        data = json.loads(line)

        data["output"] = ready_list[i]

        with open(output_file_path, "a", encoding="utf-8") as output_file:
            json.dump(data, output_file, ensure_ascii=False)
            output_file.write("\n")
        
        