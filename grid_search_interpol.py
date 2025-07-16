import kenlm
import numpy as np
from itertools import product
import argparse
from tqdm import tqdm
import time
import random

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text(text):
    return normalize_processor.tokenizer._normalize(text)

def load_models(model_paths):
    """Load all n-gram models."""
    models = {}
    for n, path in model_paths.items():
        print(f"Loading {n}-gram model from {path}")
        models[n] = kenlm.Model(path)
    return models

def load_text_data(file_path, max_samples=1000):
    """Load text data from file with format: <utterance-id> <text>"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split on first space to separate utterance-id from text
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    texts.append(parts[1])
    
    # Randomly sample max_samples if we have more data
    if len(texts) > max_samples:
        print(f"Loaded {len(texts)} text samples, randomly selecting {max_samples}")
        texts = random.sample(texts, max_samples)
    else:
        print(f"Loaded {len(texts)} text samples")
    
    return texts

def calculate_interpolated_score(text, models, lambdas, eos=True):
    """Calculate interpolated score for a single text."""
    total_score = 0.0
    for n in [2, 3, 4, 5, 6]:
        if n in models and n in lambdas:
            score = models[n].score(text, bos=True, eos=eos)
            total_score += lambdas[n] * score
    return total_score

def calculate_total_log_likelihood(texts, models, lambdas, eos=True):
    """Calculate total log likelihood for all texts."""
    total_ll = 0.0
    for text in texts:
        text = normalize_text(text)
        score = calculate_interpolated_score(text, models, lambdas, eos)
        total_ll += score
    return total_ll

def generate_lambda_combinations(num_models, step_size=0.1):
    """Generate all valid lambda combinations that sum to 1."""
    # Create a grid of possible values for each lambda
    possible_values = np.arange(0, 1 + step_size, step_size)
    possible_values = np.round(possible_values, 2)  # Round to avoid floating point issues
    
    valid_combinations = []
    
    # Generate all combinations
    for combo in product(possible_values, repeat=num_models):
        if abs(sum(combo) - 1.0) < 1e-10:  # Check if sum equals 1 (with tolerance)
            valid_combinations.append(combo)
    
    return valid_combinations

def grid_search(texts, models, step_size=0.1, eos=True):
    """Perform grid search to find optimal lambda values."""
    n_grams = sorted(models.keys())
    num_models = len(n_grams)
    
    print(f"Performing grid search with step size {step_size}")
    print(f"N-gram models: {n_grams}")
    print(f"Using {len(texts)} text samples")
    
    # Generate all valid lambda combinations
    lambda_combinations = generate_lambda_combinations(num_models, step_size)
    print(lambda_combinations)
    print(f"Testing {len(lambda_combinations)} lambda combinations")
    
    # Estimate time
    print("Estimating completion time...")
    start_time = time.time()
    
    # Time a small sample to estimate total time
    sample_size = min(10, len(lambda_combinations))
    for i, combo in enumerate(lambda_combinations[:sample_size]):
        lambdas = {n_grams[j]: combo[j] for j in range(num_models)}
        calculate_total_log_likelihood(texts, models, lambdas, eos)
        if i == 0:  # After first iteration, give initial estimate
            elapsed = time.time() - start_time
            total_estimated = elapsed * len(lambda_combinations)
            print(f"Estimated completion time: {total_estimated/60:.1f} minutes")
    
    # Reset timer for actual search
    start_time = time.time()
    
    best_ll = float('-inf')
    best_lambdas = None
    
    # Test each combination
    for combo in tqdm(lambda_combinations, desc="Grid Search"):
        # Create lambda dictionary
        lambdas = {n_grams[i]: combo[i] for i in range(num_models)}
        print(lambdas)

        # Calculate log likelihood
        ll = calculate_total_log_likelihood(texts, models, lambdas, eos)
        print(ll)

        # Update best if this is better
        if ll > best_ll:
            best_ll = ll
            best_lambdas = lambdas.copy()
    
    elapsed_time = time.time() - start_time
    print(f"\nGrid search completed in {elapsed_time/60:.1f} minutes")
    
    return best_lambdas, best_ll

def main():
    parser = argparse.ArgumentParser(description='Grid search for optimal n-gram interpolation weights')
    parser.add_argument('--text_file', default='/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/test_myst/text', help='Path to text file with utterance data')
    parser.add_argument('--gram2_path', default='2gram-model.bin', help='Path to 2-gram model')
    parser.add_argument('--gram3_path', default='3gram-model.bin', help='Path to 3-gram model')
    parser.add_argument('--gram4_path', default='4gram-model.bin', help='Path to 4-gram model')
    parser.add_argument('--gram5_path', default='5gram-model.bin', help='Path to 5-gram model')
    parser.add_argument('--gram6_path', default='6gram-model.bin', help='Path to 6-gram model')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of text samples to use (default: 1000)')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step size for grid search (default: 0.1')
    parser.add_argument('--eos', action='store_true', default=True, help='Include EOS in scoring (default: True)')
    parser.add_argument('--no_eos', action='store_false', dest='eos', help='Exclude EOS in scoring')
    
    args = parser.parse_args()
    
    # Set up model paths
    model_paths = {
        2: args.gram2_path,
        3: args.gram3_path,
        4: args.gram4_path,
        5: args.gram5_path,
        6: args.gram6_path
    }
    
    # Load models
    models = load_models(model_paths)
    
    # Load text data
    texts = load_text_data(args.text_file, args.max_samples)
    
    # Perform grid search
    best_lambdas, best_ll = grid_search(texts, models, args.step_size, args.eos)
    
    # Print results
    print("\n" + "="*50)
    print("GRID SEARCH RESULTS")
    print("="*50)
    print(f"Best log likelihood: {best_ll:.6f}")
    print("Optimal lambda values:")
    for n in sorted(best_lambdas.keys()):
        print(f"  {n}-gram lambda: {best_lambdas[n]:.3f}")
    
    # Verify lambdas sum to 1
    lambda_sum = sum(best_lambdas.values())
    print(f"Lambda sum: {lambda_sum:.6f}")
    
    # Save results to file
    with open('optimal_lambdas.txt', 'w') as f:
        f.write(f"Best log likelihood: {best_ll:.6f}\n")
        f.write("Optimal lambda values:\n")
        for n in sorted(best_lambdas.keys()):
            f.write(f"{n}-gram lambda: {best_lambdas[n]:.3f}\n")
        f.write(f"Lambda sum: {lambda_sum:.6f}\n")
    
    print("\nResults saved to 'optimal_lambdas.txt'")

if __name__ == "__main__":
    main()
