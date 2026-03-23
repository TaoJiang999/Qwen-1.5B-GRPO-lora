from dotenv import load_dotenv
import os
load_dotenv()


from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_id = 'Qwen/Qwen2.5-1.5B-Instruct-SFT-lora'
peft_model_id = "/home/waas/LLM/qwen/grpo_output/checkpoint-620"

base_model = AutoModelForCausalLM.from_pretrained(base_model_id,torch_dtype=torch.bfloat16)

model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

push_id = 'DEAR-Tao/Qwen2.5-1.5B-Instruct-GRPO-think-lora'
tokenizer.push_to_hub(push_id,token=os.getenv("HF_TOKEN"),private=False)
model.push_to_hub(push_id,token=os.getenv("HF_TOKEN"),private=False)

print("推送完成！")