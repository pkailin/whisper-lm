#!/usr/bin/env python3
"""
Extract transcripts from MyST dataset text files and create JSON output.
"""

import json
import os
from pathlib import Path

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

    # Convert to lowercase for matching, but preserve original case
    result = text

    for contraction, fixed in contractions.items():
        # Replace case-insensitively but preserve the case of the original
        import re

        # Find all matches with their positions
        matches = list(re.finditer(re.escape(contraction), result, re.IGNORECASE))

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

def extract_transcripts_from_file(filepath):
    """
    Extract transcripts from a text file with format: <utterance_id> <transcript>
    
    Args:
        filepath (str): Path to the text file
        
    Returns:
        list: List of transcript strings
    """
    transcripts = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Split on first whitespace to separate utterance_id from transcript
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    utterance_id = parts[0]
                    transcript = parts[1]

                    transcript = fix_contractions(transcript)
                    transcript = normalize_text(transcript)

                    transcripts.append(transcript)
                else:
                    print(f"Warning: Line {line_num} in {filepath} doesn't have expected format: {line}")
    
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return transcripts

def main():
    # Define file paths
    train_file = "/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/test_myst/text"
    #dev_file = "/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/dev_myst_nocomb/text"
    output_file = "myst_test.json"

    #train_file = "/home/klp65/rds/rds-altaslp-8YSp2LXTlkY/data/tedlium/tedlium/train/text"
    #dev_file = "/home/klp65/rds/rds-altaslp-8YSp2LXTlkY/data/tedlium/tedlium/dev/text"
    #output_file = "tedlium.json"
    
    print("Extracting transcripts from MyST dataset...")
    
    # Extract transcripts from both files
    train_transcripts = extract_transcripts_from_file(train_file)
    #dev_transcripts = extract_transcripts_from_file(dev_file)
    
    print(f"Found {len(train_transcripts)} transcripts in train file")
    #print(f"Found {len(dev_transcripts)} transcripts in dev file")
    
    # Combine all transcripts
    #all_transcripts = train_transcripts + dev_transcripts
    all_transcripts = train_transcripts 
    print(f"Total transcripts: {len(all_transcripts)}")

    
    # Create JSON structure
    json_data = []
    for transcript in all_transcripts:
        json_data.append({"text": transcript})
    
    # Write to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully created {output_file} with {len(json_data)} entries")
        
        # Show first few examples
        if json_data:
            print("\nFirst 3 examples:")
            for i, entry in enumerate(json_data[:3]):
                print(f"  {i+1}: {entry['text']}")
    
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    main()
