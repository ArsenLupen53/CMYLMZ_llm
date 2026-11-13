import re, sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()

# Satır başındaki zaman damgası: "0:3333 saniye", "1:00:091 saat 9 saniye" vb.
ts = re.compile(
    r"^(?:\d{1,2}:)?\d{1,2}:\d{2}"          # 0:33 veya 1:00:09
    r"\s*(?:\d+\s*saat)?\s*(?:\d+\s*dakika)?\s*(?:\d+\s*saniye)?\s*",
    re.M,
)
text = ts.sub("", text)

# [kahkaha], [alkış], [müzik] gibi etiketler
text = re.sub(r"\[[^\]]{1,20}\]", " ", text)

# Satırları birleştir, boşlukları düzelt
lines = [l.strip() for l in text.splitlines() if l.strip()]
text = " ".join(lines)
text = re.sub(r"\s+", " ", text).strip()

# Cümle sonlarında paragraflara böl (~her 5 cümlede bir)
sents = re.split(r"(?<=[.!?])\s+", text)
paras = [" ".join(sents[i:i+5]) for i in range(0, len(sents), 5)]

open(dst, "w", encoding="utf-8").write("\n\n".join(paras))