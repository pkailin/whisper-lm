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
with open("cmu_kids.json", "r") as f:
    data = json.load(f)

# Write as JSON Lines
with open("cmu_kids_lines.json", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

# If your file is JSON Lines (Option A):
dataset = load_dataset("json", data_files="cmu_kids_lines.json", split="train")

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
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print(dataset_splits['train'][0])
print(dataset_splits['train'][1])
print(dataset_splits['train'][2])


def tokenize_function(examples):
    inputs = tokenizer(
        examples['text'], 
        truncation=True, 
        padding=True,
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
model = AutoModelForCausalLM.from_pretrained('gpt2').to("cuda")

# FREEZE WEIGHTS
for param in model.parameters():
    param.requires_grad = False

# LoRa configuration
config = LoraConfig(
    r=8,
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

# Define training arguments
training_args = TrainingArguments(
    output_dir='./gpt_lora_comb1',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=200,
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_dir='./logs_lora_comb1',
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_pin_memory=False,
    remove_unused_columns=False,  # Keep all columns
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
trainer.train()

# Save the model and tokenizer explicitly
model_output_dir = './gpt_lora_comb1'

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
