#!/usr/bin/env python3
"""
Convert JSON text data to KenLM language model training pipeline
"""

import json
import subprocess
import os
import sys
from pathlib import Path

import re

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text(text):
    return normalize_processor.tokenizer._normalize(text)

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
        "doesn t": "doesn't",
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

def extract_text_from_json(json_file_path, output_text_file):
    """
    Extract text entries from JSON file and save as plain text corpus
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(output_text_file, 'w', encoding='utf-8') as f:
            for entry in data:
                if 'text' in entry and entry['text'].strip():
                    # Write each text entry on a new line
                    f.write(entry['text'].strip() + '\n')
        
        print(f"Extracted {len(data)} text entries to {output_text_file}")
        return len(data)
    
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        return 0

def preprocess_text(input_file, output_file):
    """
    Basic text preprocessing for KenLM training
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    
                    # Basic cleaning: lowercase, strip whitespace
                    text = normalize_text(fix_contractions(line.strip().lower()))
                    # Remove content within parentheses
                    text = re.sub(r'\([^)]*\)', '', text)
                    # Remove content within angle brackets
                    text = re.sub(r'<[^>]*>', '', text)
                
                    if text:  # Skip empty lines
                        f_out.write(text + '\n')
        
        print(f"Preprocessed text saved to {output_file}")
    
    except Exception as e:
        print(f"Error preprocessing text: {e}")

def train_kenlm_model(corpus_file, output_model, n_gram=3):
    """
    Train KenLM language model using lmplz
    """
    try:
        # KenLM lmplz command
        # lmplz -o 4 -S 2G --text corpus_clean.txt --arpa 4gram-model.arpa
        cmd = [
            "lmplz",
            "-o", str(n_gram),  # n-gram order
            "-S", "2G", # memory limit
            "--text", corpus_file,
            "--arpa", output_model
        ]
        
        print(f"Training {n_gram}-gram model with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully trained KenLM model: {output_model}")
            return True
        else:
            print(f"Error training model: {result.stderr}")
            return False
    
    except FileNotFoundError:
        print("Error: lmplz not found. Please ensure KenLM is installed and in PATH.")
        return False
    except Exception as e:
        print(f"Error running lmplz: {e}")
        return False

def convert_to_binary(arpa_file, binary_file):
    """
    Convert ARPA model to binary format for faster loading
    """
    try:
        cmd = ["build_binary", arpa_file, binary_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Binary model saved to: {binary_file}")
            return True
        else:
            print(f"Error converting to binary: {result.stderr}")
            return False
    
    except FileNotFoundError:
        print("build_binary not found. Binary conversion skipped.")
        return False
    except Exception as e:
        print(f"Error converting to binary: {e}")
        return False

def main():
    # Configuration
    json_input = "./train_gpt/combined5_filtered.json"  # Your JSON file path
    text_corpus = "corpus_comb5.txt"
    preprocessed_corpus = "corpus_clean_comb5.txt"
    arpa_model = "2gram-model.arpa"
    binary_model = "2gram-model.bin"
    n_gram_order = 2 # 6-gram model
    
    print("=== JSON to KenLM Training Pipeline ===")
    
    # Step 1: Extract text from JSON
    print("\n1. Extracting text from JSON...")
    if not os.path.exists(json_input):
        print(f"Error: JSON file '{json_input}' not found!")
        print("Please update the json_input variable with your file path.")
        return
    
    num_entries = extract_text_from_json(json_input, text_corpus)
    if num_entries == 0:
        print("No text entries found. Exiting.")
        return
    
    # Step 2: Preprocess text
    print("\n2. Preprocessing text...")
    preprocess_text(text_corpus, preprocessed_corpus)
    """
    # Step 3: Train KenLM model
    print(f"\n3. Training {n_gram_order}-gram KenLM model...")
    success = train_kenlm_model(preprocessed_corpus, arpa_model, n_gram_order)
    
    if not success:
        print("Training failed. Please check KenLM installation.")
        return
    
    # Step 4: Convert to binary (optional, for faster loading)
    print("\n4. Converting to binary format...")
    convert_to_binary(arpa_model, binary_model)
    
    print("\n=== Training Complete ===")
    print(f"ARPA model: {arpa_model}")
    print(f"Binary model: {binary_model}")
    print("\nUsage example:")
    print(f"query -v summary {binary_model} < test_sentences.txt")
    """
if __name__ == "__main__":
    main()


# Alternative: Simple extraction script
def simple_extract(json_file, output_file):
    """
    Simple function to just extract text from JSON
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            if 'text' in entry:
                f.write(entry['text'] + '\n')
    
    print(f"Extracted {len(data)} entries to {output_file}")
