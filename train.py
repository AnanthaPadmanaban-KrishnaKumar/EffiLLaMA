from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np
import os
from typing import Dict, List

# Suppress warnings from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def data_ingestion(directory: str, chunk_size: int = 1500, chunk_overlap: int = 50) -> list:
    """Load and split PDF documents into chunks."""
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.page_content = doc.page_content.replace("\n", " ")
        doc.metadata["id"] = f"chunk_{i}"
        doc.metadata["text"] = doc.page_content
    return docs

def prepare_dataset(document_chunks: List) -> Dataset:
    """Prepare dataset from document chunks."""
    if not document_chunks:
        raise ValueError("Document chunks are empty")
    
    data = [{"text": chunk.page_content} for chunk in document_chunks]
    return Dataset.from_list(data)

class CustomDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator that handles variable length sequences properly."""
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Find max length in this batch
        batch_max_len = max(len(feature["input_ids"]) for feature in features)
        
        # Initialize padded batch
        batch = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }
        
        # Pad sequences to max length in batch
        for feature in features:
            input_ids = feature["input_ids"]
            padding_length = batch_max_len - len(input_ids)
            
            # Pad with pad_token_id
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            
            batch["input_ids"].append(padded_input_ids)
            batch["labels"].append(padded_input_ids.copy())  # For causal LM, labels are the same as inputs
            batch["attention_mask"].append(attention_mask)
        
        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch

def tokenize_function(examples: Dict, tokenizer) -> Dict:
    """Tokenize texts without truncation."""
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=False,
        return_attention_mask=True
    )

def prepare_training(
    model_name: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer
) -> tuple:
    """Prepare model and training components."""
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically handle multi-GPU
        torch_dtype=torch.float16  # Use fp16 for efficiency
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments optimized for multi-GPU
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=100,
        logging_steps=50,
        fp16=True,
        optim="adamw_torch",
        max_grad_norm=0.3,
        max_steps=1000,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Important for variable length sequences
    )
    
    return model, training_args

def main():
    # 1. Data ingestion
    docs = data_ingestion("datasets", chunk_size=1500, chunk_overlap=50)
    dataset = prepare_dataset(docs)
    
    # 2. Split dataset
    train_test = dataset.train_test_split(test_size=0.1)
    
    # 3. Initialize tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Update with your actual model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 4. Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 5. Tokenize datasets
    train_dataset = train_test["train"].map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=train_test["train"].column_names,
        batched=True
    )
    eval_dataset = train_test["test"].map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=train_test["test"].column_names,
        batched=True
    )
    
    # 6. Prepare training components
    model, training_args = prepare_training(
        model_name,
        train_dataset,
        eval_dataset,
        tokenizer
    )
    
    # 7. Initialize custom data collator
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 8. Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 9. Train
    try:
        trainer.train()
        # Save the final model
        trainer.save_model("final_model")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()