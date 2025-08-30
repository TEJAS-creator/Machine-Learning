from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

while True:
    # Take user input
    sentence = input("\nEnter a sentence (or type 'exit' to quit): ")
    if sentence.lower() == "exit":
        print("Exiting program...")
        break

    max_tokens = int(input("Enter the number of tokens to generate: "))

    # Encode the input
    inputs = tokenizer.encode(sentence, return_tensors="pt")

    # Generate text
    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + max_tokens,

        # 🔹 do_sample=True
        do_sample=True,

        # 🔹 top_k=50 (Top-K Sampling)
        top_k=50,

        # 🔹 top_p=0.95 (Nucleus Sampling)
        top_p=0.95,

        # 🔹 temperature=0.7
        temperature=0.7,

        # Avoids warnings about padding
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated Text:\n", generated_text)
