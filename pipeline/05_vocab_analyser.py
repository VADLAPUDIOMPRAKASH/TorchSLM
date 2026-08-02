import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/telugu.model")

print("=" * 60)
print("Vocabulary Statistics")
print("=" * 60)

print(f"Vocabulary Size : {sp.get_piece_size()}")

print("\nFirst 20 Tokens")
print("-" * 60)

for i in range(20):
    print(f"{i:4} -> {sp.id_to_piece(i)}")

print("\nLast 20 Tokens")
print("-" * 60)

vocab_size = sp.get_piece_size()

for i in range(vocab_size - 20, vocab_size):
    print(f"{i:4} -> {sp.id_to_piece(i)}")

print("\nSpecial Tokens")
print("-" * 60)

print("UNK :", sp.unk_id(), sp.id_to_piece(sp.unk_id()))
print("BOS :", sp.bos_id(), sp.id_to_piece(sp.bos_id()))
print("EOS :", sp.eos_id(), sp.id_to_piece(sp.eos_id()))

print("\nLookup Examples")
print("-" * 60)

words = [
    "నేను",
    "తెలుగు",
    "హైదరాబాద్",
    "భారత్",
    "ఉంటున్నాను"
]

for word in words:
    pieces = sp.encode(word, out_type=str)
    ids = sp.encode(word, out_type=int)

    print(f"\nWord : {word}")
    print("Pieces :", pieces)
    print("IDs    :", ids)