import kenlm
import numpy as np
from itertools import product
import argparse
from tqdm import tqdm
import time
import random

import re
import jiwer
import string
import argparse
import sys

def fix_contractions(text):
    """
    Add apostrophes to common contractions in text.

    Args:
        text (str): Input text with missing apostrophes in contractions

    Returns:
        str: Text with apostrophes added to contractions
    """
    # Dictionary of contractions without apostrophes -> with apostrophes
    contractions = {
        # Common negative contractions
        "didn t": "didn't",
        "don t": "don't",
        "won t": "won't",
        "can t": "can't",
        "isn t": "isn't",
        "aren t": "aren't",
        "wasn t": "wasn't",
        "weren t": "weren't",
        "hasn t": "hasn't",
        "haven t": "haven't",
        "hadn t": "hadn't",
        "shouldn t": "shouldn't",
        "wouldn t": "wouldn't",
        "couldn t": "couldn't",
        "mustn t": "mustn't",
        "needn t": "needn't",
        "daren t": "daren't",
        "oughtn t": "oughtn't",

        # Contractions with "will"
        "i ll": "I'll",
        "you ll": "you'll",
        "he ll": "he'll",
        "she ll": "she'll",
        "it ll": "it'll",
        "we ll": "we'll",
        "they ll": "they'll",
        "that ll": "that'll",
        "who ll": "who'll",

        # Contractions with "am/is/are"
        "i m": "I'm",
        "you re": "you're",
        "he s": "he's",
        "she s": "she's",
        "it s": "it's",
        "we re": "we're",
        "they re": "they're",
        "that s": "that's",
        "who s": "who's",
        "what s": "what's",
        "where s": "where's",
        "when s": "when's",
        "how s": "how's",
        "why s": "why's",
        "there s": "there's",
        "here s": "here's",

        # Contractions with "have"
        "i ve": "I've",
        "you ve": "you've",
        "we ve": "we've",
        "they ve": "they've",
        "could ve": "could've",
        "should ve": "should've",
        "would ve": "would've",
        "might ve": "might've",
        "must ve": "must've",

        # Contractions with "had/would"
        "i d": "I'd",
        "you d": "you'd",
        "he d": "he'd",
        "she d": "she'd",
        "it d": "it'd",
        "we d": "we'd",
        "they d": "they'd",
        "that d": "that'd",
        "who d": "who'd",

        # Other common contractions
        "let s": "let's",
        "y all": "y'all",
        "o clock": "o'clock",
    }

    # Special cases to avoid - patterns that should NOT be converted
    avoid_patterns = {
        "i d cell": True,  # Specific case mentioned
        # Add other patterns as needed
    }

    # Convert to lowercase for matching, but preserve original case
    result = text
    for contraction, fixed in contractions.items():
        # Check for special avoidance patterns first
        should_skip = False
        for avoid_pattern in avoid_patterns:
            if contraction in avoid_pattern:
                # Use word boundaries to check if this contraction is part of the avoid pattern
                pattern = r'\b' + re.escape(avoid_pattern) + r'\b'
                if re.search(pattern, result, re.IGNORECASE):
                    should_skip = True
                    break

        if should_skip:
            continue

        # Use word boundaries to ensure we only match complete contractions
        pattern = r'\b' + re.escape(contraction) + r'\b'
        matches = list(re.finditer(pattern, result, re.IGNORECASE))

        # Replace from right to left to avoid position shifts
        for match in reversed(matches):
            start, end = match.span()
            original_text = result[start:end]

            # Preserve the original case pattern
            if original_text.isupper():
                replacement = fixed.upper()
            elif original_text.istitle():
                replacement = fixed.capitalize()
            elif original_text.islower():
                replacement = fixed.lower()
            else:
                # Mixed case - try to preserve the pattern
                replacement = ""
                for i, char in enumerate(fixed):
                    if i < len(original_text):
                        if original_text[i].isupper():
                            replacement += char.upper()
                        else:
                            replacement += char.lower()
                    else:
                        replacement += char.lower()
            result = result[:start] + replacement + result[end:]

    return result

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

def generate_partial_sequences(text, min_length=1):
    """
    Generate all partial sequences (prefixes) of a text.
    This simulates the partial sequences that would be evaluated during beam search.
    """
    words = text.split()
    partial_sequences = []
    
    for i in range(min_length, len(words) + 1):
        partial_seq = ' '.join(words[:i])
        partial_sequences.append(partial_seq)
    
    return partial_sequences

def calculate_interpolated_score(text, models, lambdas, is_complete=False):
    """
    Calculate interpolated score for a sequence.
    
    Args:
        text: The sequence to score
        models: Dictionary of n-gram models
        lambdas: Dictionary of interpolation weights
        is_complete: Whether this is a complete utterance (affects EOS scoring)
    """
    total_score = 0.0
    for n in [2, 3, 4, 5, 6]:
        if n in models and n in lambdas:
            # For partial sequences, we typically don't want EOS unless it's complete
            score = models[n].score(text, bos=True, eos=is_complete)
            total_score += lambdas[n] * score
    return total_score

def calculate_cumulative_log_likelihood(texts, models, lambdas):
    """
    Calculate cumulative log likelihood using beam search style scoring.
    
    For each utterance, calculates:
    log(score("the")) + log(score("the weather")) + ... + log(score("complete utterance"))
    
    Args:
        texts: List of complete reference texts
        models: Dictionary of n-gram models
        lambdas: Dictionary of interpolation weights
    """
    total_ll = 0.0
    
    for text in texts:
        text = normalize_text(fix_contractions(text))
        
        # Calculate cumulative score: sum of all prefix scores
        utterance_score = 0.0
        partial_sequences = generate_partial_sequences(text, min_length=1)
        
        for i, partial_seq in enumerate(partial_sequences):
            # For the last sequence (complete utterance), include EOS
            is_complete = (i == len(partial_sequences) - 1)
            #is_complete = False
            partial_score = calculate_interpolated_score(partial_seq, models, lambdas, is_complete=is_complete)
            utterance_score += partial_score
        
        total_ll += utterance_score
    
    # Return average log likelihood per utterance
    return total_ll / len(texts) if texts else 0.0

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

def grid_search(texts, models, step_size=0.1):
    """
    Perform grid search to find optimal lambda values using cumulative beam search scoring.
    
    Args:
        texts: List of reference texts
        models: Dictionary of n-gram models
        step_size: Step size for lambda grid search
    """
    n_grams = sorted(models.keys())
    num_models = len(n_grams)

    print(f"Performing grid search with step size {step_size}")
    print(f"N-gram models: {n_grams}")
    print(f"Using {len(texts)} text samples")
    print("Using cumulative beam search scoring")

    # Generate all valid lambda combinations
    lambda_combinations = generate_lambda_combinations(num_models, step_size)
    print(f"Testing {len(lambda_combinations)} lambda combinations")

    # Estimate time
    print("Estimating completion time...")
    start_time = time.time()

    # Time a small sample to estimate total time
    sample_size = min(10, len(lambda_combinations))
    for i, combo in enumerate(lambda_combinations[:sample_size]):
        lambdas = {n_grams[j]: combo[j] for j in range(num_models)}
        calculate_cumulative_log_likelihood(texts, models, lambdas)
            
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

        # Calculate cumulative log likelihood
        ll = calculate_cumulative_log_likelihood(texts, models, lambdas)

        print(ll)

        # Update best if this is better
        if ll > best_ll:
            best_ll = ll
            best_lambdas = lambdas.copy()

        print('Current Best LL' + str(best_ll))
        print('Current Best Lambdas' + str(best_lambdas))
        

    elapsed_time = time.time() - start_time
    print(f"\nGrid search completed in {elapsed_time/60:.1f} minutes")

    return best_lambdas, best_ll

def main():
    parser = argparse.ArgumentParser(description='Grid search for optimal n-gram interpolation weights using cumulative beam search scoring')
    parser.add_argument('--text_file', default='/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/test_myst/text', help='Path to text file with utterance data')
    parser.add_argument('--gram2_path', default='2gram-model.bin', help='Path to 2-gram model')
    parser.add_argument('--gram3_path', default='5gram-model_comb4.bin', help='Path to 3-gram model')
    parser.add_argument('--gram4_path', default='6gram-model_comb4.bin', help='Path to 4-gram model')
    parser.add_argument('--gram5_path', default='5gram-model.bin', help='Path to 5-gram model')
    parser.add_argument('--gram6_path', default='6gram-model.bin', help='Path to 6-gram model')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of text samples to use (default: 1000)')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step size for grid search (default: 0.1)')

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
    best_lambdas, best_ll = grid_search(texts, models, args.step_size)

    # Print results
    print("\n" + "="*50)
    print("GRID SEARCH RESULTS")
    print("="*50)
    print("Scoring method: Cumulative beam search")
    print(f"Best log likelihood: {best_ll:.6f}")
    print("Optimal lambda values:")
    for n in sorted(best_lambdas.keys()):
        print(f"  {n}-gram lambda: {best_lambdas[n]:.3f}")

    # Verify lambdas sum to 1
    lambda_sum = sum(best_lambdas.values())
    print(f"Lambda sum: {lambda_sum:.6f}")

    # Save results to file
    with open('optimal_lambdas.txt', 'w') as f:
        f.write("Scoring method: Cumulative beam search\n")
        f.write(f"Best log likelihood: {best_ll:.6f}\n")
        f.write("Optimal lambda values:\n")
        for n in sorted(best_lambdas.keys()):
            f.write(f"{n}-gram lambda: {best_lambdas[n]:.3f}\n")
        f.write(f"Lambda sum: {lambda_sum:.6f}\n")

    print("\nResults saved to 'optimal_lambdas.txt'")

if __name__ == "__main__":
    main()
