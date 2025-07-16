#!/usr/bin/env python3
import os
import json
import re
from pathlib import Path

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

def has_phoneme_representations(text):
    """Check if text contains phoneme representations in /.../ format"""
    pattern = r'/[^/]+/'
    return bool(re.search(pattern, text))

def extract_cmu_kids_texts():
    # Expand the home directory path
    base_path = Path.home() / "cmu_kids_trans"

    all_texts = []
    skipped_count = 0

    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist")
        return

    print(f"Processing directory: {base_path}")

    # Find all .trn files recursively
    trn_files = list(base_path.rglob("*.trn"))

    if not trn_files:
        print("No .trn files found in the directory")
        return

    print(f"Found {len(trn_files)} .trn files")

    for trn_file in trn_files:
        try:
            with open(trn_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if content:
                # Check if content contains phoneme representations
                if has_phoneme_representations(content):
                    skipped_count += 1
                    print(f"  Skipped (contains phonemes): {trn_file}")
                    continue

                # Clean up the text
                cleaned_content = normalize_text(content)

                if cleaned_content:  # Only add non-empty content after cleaning
                    all_texts.append({"text": cleaned_content})
                    print(f"  Processed: {trn_file}")
                    # Show conversion example for first few files
                    if len(all_texts) <= 5:
                        print(f"    Original: {content}")
                        print(f"    Cleaned: {cleaned_content}")
                        print()
                else:
                    print(f"  Skipping file with no content after cleaning: {trn_file}")

        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(trn_file, 'r', encoding='latin-1') as f:
                    content = f.read().strip()

                if content:
                    # Check if content contains phoneme representations
                    if has_phoneme_representations(content):
                        skipped_count += 1
                        print(f"  Skipped (contains phonemes): {trn_file} (latin-1)")
                        continue

                    cleaned_content = normalize_text(content)

                    if cleaned_content:
                        all_texts.append({"text": cleaned_content})
                        print(f"  Processed: {trn_file} (using latin-1 encoding)")

            except Exception as e:
                print(f"  Error reading {trn_file}: {e}")

        except Exception as e:
            print(f"  Error reading {trn_file}: {e}")

    # Save to JSON file
    output_file = "cmu_kids.json"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_texts, f, indent=2, ensure_ascii=False)

        print(f"\nSuccessfully created {output_file}")
        print(f"Total entries: {len(all_texts)}")
        print(f"Files skipped (contained phonemes): {skipped_count}")
        print(f"Files processed: {len(trn_files) - skipped_count}")

        # Show a sample of the final entries
        if all_texts:
            print(f"\nSample final entries:")
            for i, entry in enumerate(all_texts[:3]):
                text_preview = entry['text'][:150] + "..." if len(entry['text']) > 150 else entry['text']
                print(f"  Entry {i+1}: {text_preview}")

    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    extract_cmu_kids_texts()
