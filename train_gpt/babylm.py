#!/usr/bin/env python3
import os
import json
from pathlib import Path

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text(text):
    return normalize_processor.tokenizer._normalize(text)

def process_babylm_data():
    # Define base directories and file patterns
    base_dirs = [
        "~/rds/rds-altaslp-8YSp2LXTlkY/data/babylm_data/babylm_data/babylm_100M",
        "~/rds/rds-altaslp-8YSp2LXTlkY/data/babylm_data/babylm_data/babylm_dev", 
        "~/rds/rds-altaslp-8YSp2LXTlkY/data/babylm_data/babylm_data/babylm_test"
    ]
    
    file_patterns = [
        "aochildes",
        "children_stories", 
        "simple_wikipedia",
        "cbt"
    ]
    
    # Extensions for each directory
    extensions = ["train", "dev", "test"]
    
    # Dictionary to store entries for each file pattern
    pattern_entries = {pattern: [] for pattern in file_patterns}
    
    for i, base_dir in enumerate(base_dirs):
        # Expand the ~ to full path
        expanded_dir = Path(base_dir).expanduser()
        extension = extensions[i]
        
        print(f"Processing directory: {expanded_dir}")
        
        if not expanded_dir.exists():
            print(f"Warning: Directory {expanded_dir} does not exist")
            continue
            
        for pattern in file_patterns:
            filename = f"{pattern}.{extension}"
            filepath = expanded_dir / filename
            
            print(f"Looking for file: {filepath}")
            
            if not filepath.exists():
                print(f"Warning: File {filepath} does not exist")
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    print(f"Processing {filepath}...")
                    line_count = 0
                    
                    for line in f:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        # Split by periods and process each segment
                        segments = line.split('.')
                        
                        for segment in segments:
                            segment = segment.strip()
                            segment = normalize_text(segment)

                            if segment:  # Only add non-empty segments
                                pattern_entries[pattern].append({"text": segment})
                                line_count += 1
                    
                    print(f"Added {line_count} text segments from {filename}")
                    
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    
    # Write separate JSON files for each pattern
    for pattern in file_patterns:
        output_file = f"{pattern}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pattern_entries[pattern], f, indent=2, ensure_ascii=False)
            
            print(f"Successfully created {output_file} with {len(pattern_entries[pattern])} entries")
            
        except Exception as e:
            print(f"Error writing to {output_file}: {e}")
    
    # Print summary
    total_entries = sum(len(entries) for entries in pattern_entries.values())
    print(f"\nTotal entries processed: {total_entries}")

if __name__ == "__main__":
    process_babylm_data()
