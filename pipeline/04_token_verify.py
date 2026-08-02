import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/telugu.model")

text = "నేను హైదరాబాద్‌లో ఉంటున్నాను. what is this"

pieces = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)

print("Text :", text)
print("Tokens:", pieces)
print("IDs   :", ids)

print("Decoded:", sp.decode(ids))