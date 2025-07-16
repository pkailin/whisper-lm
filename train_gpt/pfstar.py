#!/usr/bin/env python3
"""
PFSTAR TRS File Parser
Processes .trs files from PFSTAR dataset and generates pfstar.json
"""

import os
import json
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from collections import OrderedDict

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
    text = re.sub(r"\[[^\]]+\]", "", text) # remove words in []

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
    return normalize_processor.tokenizer._normalize(text)


def extract_age_from_filename(filename):
    """Extract age from filename like '003m13bh' -> 13"""
    match = re.search(r'(\d{3})[mf](\d{2})', filename)
    if match:
        return int(match.group(2))
    return None

def is_valid_age(age):
    """Check if age is between 6-13 inclusive"""
    
    #return age is not None and 6 <= age <= 13

    return True 

def should_skip_file(filename):
    """Check if file should be skipped (digits* or w_list*)"""
    basename = os.path.basename(filename)
    return basename.startswith('digits') or basename.startswith('w_list')

def parse_trs_file(filepath):
    """Parse a single .trs file and extract sentences"""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        sentences = []
        current_sentence = []
        
        # Find all Sync elements and text nodes
        for turn in root.findall('.//Turn'):
            for element in turn:
                if element.tag == 'Sync':
                    # Get text that follows this sync element
                    if element.tail:
                        text = element.tail.strip()
                        if text:
                            if text == 'sp':
                                # End of sentence - save current sentence if not empty
                                if current_sentence:
                                    sentence_text = ' '.join(current_sentence).strip()
                                    if sentence_text:
                                        sentences.append(sentence_text)
                                    current_sentence = []
                            elif text == 'sil':
                                # Silence - ignore
                                continue
                            else:
                                # Regular text - add to current sentence
                                # Clean up text (remove brackets and content inside)
                                cleaned_text = normalize_text(text)
                                if cleaned_text:
                                    current_sentence.append(cleaned_text)
                
                # Also check for direct text content in Turn
                if element.tag != 'Sync' and element.tag != 'Event':
                    if element.text:
                        text = element.text.strip()
                        if text and text not in ['sp', 'sil']:
                            cleaned_text = normalize_text(text)
                            if cleaned_text:
                                current_sentence.append(cleaned_text)
        
        # Don't forget the last sentence if it doesn't end with 'sp'
        if current_sentence:
            sentence_text = ' '.join(current_sentence).strip()
            if sentence_text:
                sentences.append(sentence_text)
        
        return sentences
        
    except ET.ParseError as e:
        print(f"Error parsing {filepath}: {e}")
        return []
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []

def process_pfstar_dataset(base_path):
    """Process the entire PFSTAR dataset"""
    base_path = Path(base_path)
    all_sentences = []
    processed_files = 0
    skipped_files = 0
    
    print("Processing PFSTAR dataset...")
    
    # Iterate through all speaker directories
    for speaker_dir in sorted(base_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
            
        # Extract age from directory name
        age = extract_age_from_filename(speaker_dir.name)
        
        if not is_valid_age(age):
            print(f"Skipping {speaker_dir.name} (age {age} not in range 6-13)")
            continue
            
        print(f"Processing {speaker_dir.name} (age {age})...")
        
        # Process both 'adapt' and 'test' subdirectories
        for subdir_name in ['adapt', 'test']:
            subdir = speaker_dir / subdir_name
            if not subdir.exists():
                continue
                
            # Process all .trs files in subdirectory
            for trs_file in subdir.glob('*.trs'):
                if should_skip_file(trs_file.name):
                    print(f"  Skipping {trs_file.name} (starts with digits or w_list)")
                    skipped_files += 1
                    continue
                    
                print(f"  Processing {trs_file.name}")
                sentences = parse_trs_file(trs_file)
                all_sentences.extend(sentences)
                processed_files += 1
    
    print(f"\nProcessed {processed_files} files, skipped {skipped_files} files")
    print(f"Extracted {len(all_sentences)} sentences total")
    
    # Remove duplicates while preserving order
    unique_sentences = list(OrderedDict.fromkeys(all_sentences))
    print(f"After removing duplicates: {len(unique_sentences)} unique sentences")
    
    return unique_sentences

def create_json_output(sentences, output_file):
    """Create the JSON output file"""
    json_data = []
    for sentence in sentences:
        if sentence.strip():  # Only add non-empty sentences
            json_data.append({"text": sentence.strip()})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(json_data)} entries to {output_file}")

def main():
    # Configuration
    base_path = os.path.expanduser("~/pfstar_trans")
    output_file = "pfstar_unfiltered.json"
    
    print(f"Base path: {base_path}")
    print(f"Output file: {output_file}")
    
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist")
        return
    
    # Process the dataset
    sentences = process_pfstar_dataset(base_path)
    
    # Create JSON output
    create_json_output(sentences, output_file)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
