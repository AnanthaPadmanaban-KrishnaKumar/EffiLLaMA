import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model_and_tokenizer(model_name: str, lora_checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, lora_checkpoint)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer

def generate_response(
    model,
    tokenizer,
    input_text: str,
    max_length: int = 512,
    num_beams: int = 4,
    temperature: float = 0.7,  # Effective only if do_sample=True
    top_k: int = 50,
    top_p: float = 0.9,        # Effective only if do_sample=True
    repetition_penalty: float = 1.2,
    do_sample: bool = True     # Enable sampling to match other parameters
):
    prompt_template = """Analyze the given question based on facts established in the Harry Potter series canon.

Rules:
1. Use only information from the books, films, or official sources like interviews with J.K. Rowling.
2. Avoid inventing details, characters, or events not present in canon.
3. If analysis or interpretation is provided, explicitly state it as such.

Question: {input_text}

Factual analysis:"""
    
    inputs = tokenizer(
        prompt_template.format(input_text=input_text),
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace(prompt_template.format(input_text=input_text), "").strip()

def main():
    # Configuration
    BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    LORA_CHECKPOINT = "./final_model"

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_NAME, LORA_CHECKPOINT)
    print("Model and tokenizer loaded successfully!")

    input_prompts = [
        "Why do you think the Elder Wand, despite being the most powerful wand, ultimately caused so much trouble for its owners? What does this tell us about power and responsibility in the wizarding world?"
    ]

    for i, prompt in enumerate(input_prompts, 1):
        print(f"\nInput Prompt {i}:")
        print(prompt)
        response = generate_response(model, tokenizer, prompt)
        print("\nGenerated Response:")
        print(response)
        print("-" * 80)

if __name__ == "__main__":
    main()

