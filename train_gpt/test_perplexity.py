import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
import math

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

def calculate_dataset_perplexity(model, tokenizer, utterances, device, max_length=1024):
    """Calculate average perplexity across all utterances in dataset"""
    perplexities = []
    failed_count = 0
    
    print(f"Calculating perplexity for {len(utterances)} utterances...")
    
    for utterance_id, text in tqdm(utterances):
        perplexity = calculate_perplexity(model, tokenizer, text, device, max_length)
        
        if perplexity is not None and not math.isinf(perplexity) and not math.isnan(perplexity):
            perplexities.append(perplexity)
        else:
            failed_count += 1
    
    if not perplexities:
        print("Error: No valid perplexity scores calculated!")
        return None, None, None
    
    avg_perplexity = np.mean(perplexities)
    median_perplexity = np.median(perplexities)
    std_perplexity = np.std(perplexities)
    
    print(f"Successfully calculated perplexity for {len(perplexities)} utterances")
    if failed_count > 0:
        print(f"Failed to process {failed_count} utterances")
    
    return avg_perplexity, median_perplexity, std_perplexity

def main():
    parser = argparse.ArgumentParser(description='Compare perplexity of two language models on a text dataset')
    parser.add_argument('--model1', required=True, help='Path to first language model (HuggingFace model name or local path)')
    parser.add_argument('--model2', required=True, help='Path to second language model (HuggingFace model name or local path)')
    parser.add_argument('--dataset', required=True, help='Path to text dataset file')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length for tokenization')
    parser.add_argument('--model1_name', default='Model 1', help='Display name for first model')
    parser.add_argument('--model2_name', default='Model 2', help='Display name for second model')
    
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
    avg_ppl1, med_ppl1, std_ppl1 = calculate_dataset_perplexity(
        model1, tokenizer1, utterances, device, args.max_length
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
    model2, tokenizer2, device = load_model_and_tokenizer(args.model2)
    
    # Calculate perplexity for second model
    print(f"\n=== {args.model2_name} ===")
    avg_ppl2, med_ppl2, std_ppl2 = calculate_dataset_perplexity(
        model2, tokenizer2, utterances, device, args.max_length
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

if __name__ == "__main__":
    main()
