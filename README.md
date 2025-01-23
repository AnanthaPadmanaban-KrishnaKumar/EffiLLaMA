# EffiLLaMA

## Overview
This project demonstrates how to fine-tune the LLaMA 3.2-1B Instruct model using text extracted from the Harry Potter book series. The training was conducted using LoRA (Low-Rank Adaptation) and QLoRA techniques for parameter-efficient fine-tuning. The goal is to create a custom model that generates Harry Potter-themed text and understands the context specific to the book series.


## Key Features
1. Extracted text from all Harry Potter books and chunked it for training.
2. Fine-tuned LLaMA 3.2-1B using LoRA and QLoRA for causal language modeling.
3. Efficient parameter tuning with reduced memory requirements.
4. Saved fine-tuned model weights for inference or further fine-tuning.


## What is LoRA?
LoRA (Low-Rank Adaptation) is a technique designed to make fine-tuning large language models more efficient. Instead of updating all model parameters during fine-tuning, LoRA introduces additional low-rank trainable matrices into specific layers of the model, significantly reducing the number of parameters that need to be updated.

### Benefits of LoRA:
1. Significantly reduces memory usage.
2. Faster fine-tuning on large-scale models.
3. Maintains high performance with fewer trainable parameters.

For a deeper understanding, refer to the LoRA paper:

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by Edward J. Hu et al.

## What is QLoRA?
QLoRA (Quantized LoRA) builds upon the LoRA framework by using quantization techniques. It leverages 4-bit quantization for the model weights to further reduce memory usage while maintaining the ability to fine-tune efficiently.

### QLoRA uses:
1. 4-bit quantized base models for reduced memory consumption.
2. Parameter-efficient fine-tuning to adapt the model to new tasks.
### Key Advantages:
1. Allows training of large-scale models on a single GPU.
2. Reduces the computational footprint without sacrificing performance.

For more details, check the QLoRA paper:

[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) by Tim Dettmers et al.

## Setup Instructions
### Step 1: Clone the Repository

```
git clone https://github.com/AnanthaPadmanaban-KrishnaKumar/EffiLLaMA.git
cd EffiLLaMA
```

### Step 2: Set Up a Virtual Environment (Recommended)
```
python -m venv env
source env/bin/activate   # On Linux/macOS
env\Scripts\activate      # On Windows
```

### Step 3: Install Dependencies
```
pip install -r requirements.txt
```

## Dataset Preparation
The dataset is based on the text extracted from the Harry Potter book series. The preprocessing steps included:

1. Loading PDF files using PyPDFDirectoryLoader.
2. Splitting text into chunks using RecursiveCharacterTextSplitter with:
    Chunk Size: 1500 tokens
    Chunk Overlap: 50 tokens
3. Normalizing the text (e.g., removing unnecessary characters and newlines).

## Dataset Format

The resulting dataset is stored as a list of dictionaries with the format:

```
[
  {
    "text": "Harry Potter and the Philosopher's Stone begins with..."
  },
  ...
]
```


## Training the Model

The train.py script implements the following:

1. Data Preprocessing: Loads, chunks, and normalizes the text.
2. Dataset Preparation: Converts text into a Hugging Face Dataset for training.
3. Model Initialization: Loads the LLaMA 3.2-1B Instruct model and tokenizer.
4. LoRA Configuration: Applies parameter-efficient tuning with LoRA.
5. Training: Fine-tunes the model using Trainer with mixed precision (fp16)

## Run Training
```
python train.py
```
## LoRA Configuration

```
lora_config = LoraConfig(
    r=16,                        
    lora_alpha=32,               
    target_modules=[             
        "q_proj", "v_proj", 
        "k_proj", "o_proj", 
        "gate_proj", "up_proj", 
        "down_proj"
    ],
    lora_dropout=0.1,            
    bias="none",                 
    task_type="CAUSAL_LM"        
)
```
### Key Parameters:
1. r: Low-rank dimension for decomposition matrices.
2. lora_alpha: Scaling factor for LoRA outputs.
3. target_modules: Specifies which layers to adapt with LoRA.
4. lora_dropout: Dropout rate for regularization.
5. task_type: Task type set to CAUSAL_LM (causal language modeling).
### Training Output
1. The fine-tuned model is saved in the final_model directory.
2. Logs and checkpoints are saved during training for monitoring progress.

## Acknowledgments
Hugging Face Transformers for tools to load and fine-tune the model.
LangChain for efficient text preprocessing.
Research on LoRA and QLoRA for parameter-efficient fine-tuning methods.

## License
This project is licensed under the MIT. See the [LICENSE](https://github.com/AnanthaPadmanaban-KrishnaKumar/EffiLLaMA/blob/main/LICENSE) file for details.
