import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use GPU 0


from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json

import torch
import transformers
from peft import LoraConfig, get_peft_model

# Load the JSON array
with open("combined5_filtered.json", "r") as f:
    data = json.load(f)

# Write as JSON Lines
with open("combined5_lines.json", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

# If your file is JSON Lines (Option A):
dataset = load_dataset("json", data_files="combined5_lines.json", split="train")

# You only need the 'text' field from your dataset.
# Remember to normalise the transcripts before including it in the dataset
print(dataset)

# Set a random seed for reproducibility
np.random.seed(42)

# Define train/validation split ratio
train_ratio = 0.8  # 80% training, 20% validation

# Get indices for splitting
dataset_size = len(dataset)
indices = np.random.permutation(dataset_size)
train_size = int(train_ratio * dataset_size)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create the splits
train_dataset = dataset.select(train_indices)
val_dataset = dataset.select(val_indices)

# Combine into a DatasetDict for easier handling
dataset_splits = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# Check the sizes
print(f"Total dataset size: {dataset_size}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('phonemetransformers/GPT2-85M-BPE-TXT')
tokenizer.pad_token = tokenizer.eos_token

print(dataset_splits['train'][0])
print(dataset_splits['train'][1])
print(dataset_splits['train'][2])


def tokenize_function(examples):
    inputs = tokenizer(
        examples['text'], 
        truncation=True, 
        padding="max_length",
        max_length=128,
    )
    # Set labels to be the same as input_ids for causal language modeling
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Apply tokenization
tokenized_datasets = dataset_splits.map(tokenize_function, batched=True)

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load tokenizer and model
model = AutoModelForCausalLM.from_pretrained('phonemetransformers/GPT2-85M-BPE-TXT').to("cuda")

# FREEZE WEIGHTS
for param in model.parameters():
    param.requires_grad = False

# LoRa configuration
config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)

training_args = TrainingArguments(
    output_dir='./babylm_lora_comb5',
    eval_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=15,
    per_device_train_batch_size=16,  # REDUCED from 32 - smaller batches = more stable gradients
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # MAINTAIN effective batch size of 32
    warmup_steps=800,  # INCREASED warmup for more gradual LR ramp
    learning_rate=5e-4,  # FURTHER REDUCED learning rate
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,  # MORE AGGRESSIVE gradient clipping
    logging_dir='./logs_lora_babylm',
    logging_strategy="steps",
    logging_steps=50,  # More frequent logging to monitor
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    # Stability improvements
    fp16=True,
    dataloader_drop_last=True,
    # Additional oscillation reduction techniques
    save_steps=500,  # Save more frequently
    eval_steps=500,   # Evaluate more frequently
    seed=42,          # Fixed seed for reproducibility
    data_seed=42,     # Fixed data seed
    optim="adamw_torch",           # Use PyTorch's AdamW (often more stable)
    adam_beta1=0.9,                # Default beta1 for AdamW
    adam_beta2=0.98,               # Slightly higher beta2 for more stability (default is 0.999)
    adam_epsilon=1e-6,             # Slightly smaller epsilon for numerical stability
    #resume_from_checkpoint='./gpt_lora_comb1/checkpoint-12240',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    #data_collator=data_collator
)

# Train the model
trainer.train(resume_from_checkpoint='./babylm_lora_comb5/checkpoint-47172')

# Save the model and tokenizer explicitly
model_output_dir = './babylm_lora_comb5'

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
