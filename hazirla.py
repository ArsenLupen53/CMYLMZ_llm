# hazirla.py
import json
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B"
tok = AutoTokenizer.from_pretrained(MODEL)

text = open("temiz.txt", encoding="utf-8").read()
ids = tok(text, add_special_tokens=False)["input_ids"]

SIZE, STRIDE = 512, 384          # %25 örtüşme
chunks = []
for i in range(0, len(ids), STRIDE):
    part = ids[i:i + SIZE]
    if len(part) < 64:
        break
    chunks.append(tok.decode(part))

# Örtüşme olduğu için val'i rastgele değil, sondan ayır (sızıntı olmasın)
cut = int(len(chunks) * 0.9)
for name, data in [("train.jsonl", chunks[:cut]), ("val.jsonl", chunks[cut + 8:])]:
    with open(name, "w", encoding="utf-8") as f:
        for c in data:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")

print(len(chunks), "parça")