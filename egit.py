# egit.py
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "Qwen/Qwen2.5-1.5B"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

ds = load_dataset("json", data_files={"train": "train.jsonl", "val": "val.jsonl"})
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=512),
            batched=True, remove_columns=["text"])

args = TrainingArguments(
    output_dir="out", num_train_epochs=3, per_device_train_batch_size=4,
    gradient_accumulation_steps=4, learning_rate=2e-4, warmup_ratio=0.05,
    lr_scheduler_type="cosine", fp16=True, logging_steps=5,
    eval_strategy="epoch", save_strategy="epoch", report_to="none")

Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

model.save_pretrained("lora-out")
tok.save_pretrained("lora-out")