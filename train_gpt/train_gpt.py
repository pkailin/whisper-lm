from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import json

# Load the JSON array
with open("combined3_filtered.json", "r") as f:
    data = json.load(f)

# Write as JSON Lines
with open("combined3_lines.json", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

# If your file is JSON Lines (Option A):
dataset = load_dataset("json", data_files="combined3_lines.json", split="train")

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

# input_ids is the numerical representation of your text after tokenization.
# By setting labels = input_ids, you're telling the model "your target output should be the same as your input".

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

tokenized_datasets = dataset_splits.map(tokenize_function, batched=True)

tokenizer.pad_token = tokenizer.eos_token

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to("cuda")


# Define training arguments
training_args = TrainingArguments(
    output_dir='./gpt_syn_corpora',
    evaluation_strategy='epoch',
    save_strategy = 'epoch',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_syn',
    # Add these new parameters:
    logging_strategy="steps",       # Log training metrics by steps
    logging_steps=100,              # Log every 100 steps
    report_to="tensorboard",              # Enable wandb reporting
    load_best_model_at_end=True,        # Optional: load best model at end
    metric_for_best_model="eval_loss",# Optional: load the best model when done
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Train the model
trainer.train()

# save the model and tokenizer explicitly
model_output_dir = './gpt_model'

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
