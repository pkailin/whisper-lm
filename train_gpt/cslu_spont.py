#!/usr/bin/env python3
import os
import json
from pathlib import Path

import re

"""
from whisper_normalizer.english import EnglishTextNormalizer
from num2words import num2words

# Initialize the Whisper text normalizer
normalizer = EnglishTextNormalizer()

def normalize_text(text):
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Apply Whisper's normalizer
    text = normalizer(text)

    # Step 3: Remove all characters except letters, numbers, and spaces
    text = re.sub(r"<[^>]+>", "", text) # remove words in <>

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove section from 'a b c' to the first occurrence of 'z'
    text = re.sub(r"a b c.*?z", "", text)

    # Step 4: Convert numbers into text
    # First, handle ordinal numbers like 1st, 2nd, 3rd, 4th
    def replace_ordinal(match):
        num_str = match.group(1)
        try:
            return num2words(int(num_str), ordinal=True)
        except ValueError:
            return match.group(0)

    # Handle regular numbers
    def replace_number(match):
        num = match.group(0)

        # Handle special cases like decimal numbers
        if '.' in num:  # Handle decimal numbers
            try:
                return num2words(float(num))
            except ValueError:
                return num
        else:
            try:
                return num2words(int(num))
            except ValueError:
                return num

    # First replace ordinals (must be done before regular numbers)
    # Pattern for ordinals like 1st, 2nd, 3rd, 4th, etc.
    ordinal_pattern = r'\b(\d+)(st|nd|rd|th)\b'
    text = re.sub(ordinal_pattern, replace_ordinal, text)

    # Then replace regular numbers
    # Pattern for regular numbers
    number_pattern = r'\b\d+\b|\b\d+\.\d+\b'
    text = re.sub(number_pattern, replace_number, text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

"""

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text(text):
    text = re.sub(r"a b c.*?z", "", text)
    return normalize_processor.tokenizer._normalize(text)

def extract_cslu_texts():
    base_path = "/home/klp65/rds/rds-altaslp-8YSp2LXTlkY/data/cslu_kids/trans/spontaneous"
    
    # Directories to process (01 through 08)
    #target_dirs = [f"{i:02d}" for i in range(1, 9)]

    # Get all subdirectories (no age limit) 
    base_dir = Path(base_path)
    target_dirs = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    all_texts = []
    
    for dir_num in target_dirs:
        dir_path = Path(base_path) / dir_num
        
        if not dir_path.exists():
            print(f"Warning: Directory {dir_path} does not exist, skipping...")
            continue
            
        print(f"Processing directory: {dir_path}")
        
        # Walk through all subdirectories and find .txt files
        for txt_file in dir_path.rglob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content:  # Only add non-empty content
                    content = normalize_text(content)
                    all_texts.append({"text": content})
                    print(f"  Extracted text from: {txt_file}")
                else:
                    print(f"  Skipping empty file: {txt_file}")
                    
            except Exception as e:
                print(f"  Error reading {txt_file}: {e}")
    
    # Save to JSON file
    output_file = "cslu_spont_unfiltered.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_texts, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully created {output_file}")
        print(f"Total entries: {len(all_texts)}")
        
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    extract_cslu_texts()
