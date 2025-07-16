import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
import math
from peft import PeftModel

def load_model_and_tokenizer(model_path):
    """Load language model and tokenizer from HuggingFace path or local directory"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set model to evaluation mode
    model.eval()

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model, tokenizer, device

def load_model_peft(model_path):
    #base_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
    base_model = AutoModelForCausalLM.from_pretrained('phonemetransformers/GPT2-85M-BPE-TXT')

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)

    # Merge adapter for faster inference
    model = model.merge_and_unload()
    print("LoRA adapter merged with base model")

    #tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer = AutoTokenizer.from_pretrained('phonemetransformers/GPT2-85M-BPE-TXT')

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model, tokenizer, device

def calculate_perplexity(model, tokenizer, text, device, max_length=1024):
    """Calculate perplexity for a single text"""
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)

    # Skip if text is too short
    if input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        # Get model outputs
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        # Convert loss to perplexity
        perplexity = torch.exp(loss).item()

    return perplexity

def load_dataset(file_path):
    """Load dataset from file with format: <utterance_id> <text>"""
    utterances = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Split on first space to separate ID from text
            parts = line.split(' ', 1)
            if len(parts) < 2:
                print(f"Warning: Line {line_num} doesn't have proper format, skipping")
                continue

            utterance_id = parts[0]
            text = parts[1]
            utterances.append((utterance_id, text))

    return utterances

# NEW FUNCTION: Print highest perplexity utterances
def print_highest_perplexities(perplexity_data, model_name, top_n=10):
    """Print the utterances with highest perplexities"""
    # Sort by perplexity (descending)
    sorted_data = sorted(perplexity_data, key=lambda x: x[2], reverse=True)
    
    print(f"\n=== TOP {top_n} HIGHEST PERPLEXITY UTTERANCES FOR {model_name} ===")
    for i, (utterance_id, text, perplexity) in enumerate(sorted_data[:top_n], 1):
        print(f"\n{i}. ID: {utterance_id}")
        print(f"   Perplexity: {perplexity:.4f}")
        print(f"   Text: {text[:200]}{'...' if len(text) > 200 else ''}")  # Limit text display
        print("-" * 80)

def calculate_dataset_perplexity(model, tokenizer, utterances, device, max_length=1024, model_name="Model"):
    """Calculate average perplexity across all utterances in dataset"""
    perplexities = []
    perplexity_data = []  # NEW: Store (id, text, perplexity) tuples
    failed_count = 0

    print(f"Calculating perplexity for {len(utterances)} utterances...")

    for utterance_id, text in tqdm(utterances):
        perplexity = calculate_perplexity(model, tokenizer, text, device, max_length)

        if perplexity is not None and not math.isinf(perplexity) and not math.isnan(perplexity):
            perplexities.append(perplexity)
            perplexity_data.append((utterance_id, text, perplexity))  # NEW: Store detailed data
        else:
            failed_count += 1

    if not perplexities:
        print("Error: No valid perplexity scores calculated!")
        return None, None, None, []

    avg_perplexity = np.mean(perplexities)
    median_perplexity = np.median(perplexities)
    std_perplexity = np.std(perplexities)

    print(f"Successfully calculated perplexity for {len(perplexities)} utterances")
    if failed_count > 0:
        print(f"Failed to process {failed_count} utterances")

    # NEW: Print highest perplexity utterances
    print_highest_perplexities(perplexity_data, model_name)

    return avg_perplexity, median_perplexity, std_perplexity, perplexity_data  # NEW: Return detailed data

def main():
    parser = argparse.ArgumentParser(description='Compare perplexity of two language models on a text dataset')
    parser.add_argument('--model1', required=True, help='Path to first language model (HuggingFace model name or local path)')
    parser.add_argument('--model2', required=True, help='Path to second language model (HuggingFace model name or local path)')
    parser.add_argument('--dataset', required=True, help='Path to text dataset file')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length for tokenization')
    parser.add_argument('--model1_name', default='Model 1', help='GPT Baseline')
    parser.add_argument('--model2_name', default='Model 2', help='GPT PEFT')
    parser.add_argument('--top_n', type=int, default=10, help='Number of highest perplexity utterances to display')  # NEW ARGUMENT

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    utterances = load_dataset(args.dataset)
    print(f"Loaded {len(utterances)} utterances")

    if len(utterances) == 0:
        print("Error: No utterances loaded from dataset!")
        return
    
    # Load first model
    print(f"\nLoading {args.model1_name} from {args.model1}...")
    model1, tokenizer1, device = load_model_and_tokenizer(args.model1)

    # Calculate perplexity for first model
    print(f"\n=== {args.model1_name} ===")
    avg_ppl1, med_ppl1, std_ppl1, data1 = calculate_dataset_perplexity(  # NEW: Capture detailed data
        model1, tokenizer1, utterances, device, args.max_length, args.model1_name
    )

    if avg_ppl1 is not None:
        print(f"Average Perplexity: {avg_ppl1:.4f}")
        print(f"Median Perplexity: {med_ppl1:.4f}")
        print(f"Std Dev Perplexity: {std_ppl1:.4f}")

    # Clear first model from memory
    del model1, tokenizer1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load second model
    print(f"\nLoading {args.model2_name} from {args.model2}...")
    model2, tokenizer2, device = load_model_peft(args.model2)

    # Calculate perplexity for second model
    print(f"\n=== {args.model2_name} ===")
    avg_ppl2, med_ppl2, std_ppl2, data2 = calculate_dataset_perplexity(  # NEW: Capture detailed data
        model2, tokenizer2, utterances, device, args.max_length, args.model2_name
    )

    if avg_ppl2 is not None:
        print(f"Average Perplexity: {avg_ppl2:.4f}")
        print(f"Median Perplexity: {med_ppl2:.4f}")
        print(f"Std Dev Perplexity: {std_ppl2:.4f}")

    # Comparison
    if avg_ppl1 is not None and avg_ppl2 is not None:
        print(f"\n=== COMPARISON ===")
        print(f"{args.model1_name}: {avg_ppl1:.4f} (avg), {med_ppl1:.4f} (median)")
        print(f"{args.model2_name}: {avg_ppl2:.4f} (avg), {med_ppl2:.4f} (median)")

        if avg_ppl1 < avg_ppl2:
            improvement = ((avg_ppl2 - avg_ppl1) / avg_ppl2) * 100
            print(f"{args.model1_name} has {improvement:.2f}% lower perplexity (better)")
        elif avg_ppl2 < avg_ppl1:
            improvement = ((avg_ppl1 - avg_ppl2) / avg_ppl1) * 100
            print(f"{args.model2_name} has {improvement:.2f}% lower perplexity (better)")
        else:
            print("Both models have similar perplexity")

        # NEW: Print side-by-side comparison of problematic utterances
        print(f"\n=== SIDE-BY-SIDE COMPARISON OF PROBLEMATIC UTTERANCES ===")
        if data1 and data2:
            # Find utterances that are problematic for both models
            data1_dict = {uid: (text, ppl) for uid, text, ppl in data1}
            data2_dict = {uid: (text, ppl) for uid, text, ppl in data2}
            
            # Get top problematic utterances from model 1
            sorted_data1 = sorted(data1, key=lambda x: x[2], reverse=True)
            
            print(f"\nTop {min(5, len(sorted_data1))} utterances with highest perplexity for {args.model1_name}:")
            for i, (uid, text, ppl1) in enumerate(sorted_data1[:5], 1):
                ppl2 = data2_dict.get(uid, (None, None))[1]
                print(f"\n{i}. ID: {uid}")
                print(f"   {args.model1_name} Perplexity: {ppl1:.4f}")
                print(f"   {args.model2_name} Perplexity: {ppl2:.4f}" if ppl2 else "   Model 2: N/A")
                print(f"   Text: {text[:150]}{'...' if len(text) > 150 else ''}")

if __name__ == "__main__":
    main()
