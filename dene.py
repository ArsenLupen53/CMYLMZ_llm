# dene.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "lora-out")
tok = AutoTokenizer.from_pretrained("lora-out")

prompt = "Askerde ilk günüm"
x = tok(prompt, return_tensors="pt").to(model.device)
y = model.generate(**x, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.9)
print(tok.decode(y[0], skip_special_tokens=True))