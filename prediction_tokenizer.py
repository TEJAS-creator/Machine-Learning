from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Put model in evaluation mode
model.eval()
# Clean printing function
def clean_token(token):
    # Replace invisible or weird tokens
    return repr(token).strip("'")

while True:
    sentence = input("\nEnter your sentence (or type 'exit' to quit): ")
    if sentence.lower() == "exit":
        break

    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    last_token_logits = predictions[0, -1, :]
    probs = torch.softmax(last_token_logits, dim=-1)

    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)
    predicted_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    print(f"\nInput: {sentence}")
    print("\nTop predictions for the next token:")
    for token, prob in zip(predicted_tokens, top_probs):
        print(f"{clean_token(token)}  -->  {prob.item()*100:.2f}%")
