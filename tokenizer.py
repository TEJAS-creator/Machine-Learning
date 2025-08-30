from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

sentence = input("Enter a sentence: ")

# Tokenize
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n🔹 Tokens:", tokens)
print("🔹 Token IDs:", token_ids)
print("🔹 Number of tokens:", len(tokens))
