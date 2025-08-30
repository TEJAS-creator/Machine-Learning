from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load pre-trained GPT-Neo 1.3B
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()  # set model to evaluation mode

while True:
    # Take user input
    sentence = input("\nEnter a sentence (or type 'exit' to quit): ")
    if sentence.lower() == "exit":
        print("Exiting program...")
        break

    max_tokens = int(input("Enter the number of tokens to generate: "))

    # Encode the input
    inputs = tokenizer(sentence, return_tensors="pt")

    # Generate text
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated Text:\n", generated_text)
