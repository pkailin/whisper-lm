#!/usr/bin/env python3
"""
Filter utterances to remove those with less than 3 words
"""

import json

def filter_utterances(input_file, output_file, min_words=3):
    """
    Filter utterances to keep only those with at least min_words words.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        min_words (int): Minimum number of words required
    """
    try:
        # Load the data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} utterances from {input_file}")
        
        # Filter utterances
        filtered_data = []
        removed_count = 0
        
        for entry in data:
            text = entry.get('text', '')
            word_count = len(text.split())
            
            if word_count >= min_words:
                filtered_data.append(entry)
            else:
                removed_count += 1
        
        # Save filtered data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print(f"\nFiltering complete!")
        print(f"Original utterances: {len(data):,}")
        print(f"Filtered utterances: {len(filtered_data):,}")
        print(f"Removed utterances: {removed_count:,}")
        print(f"Retention rate: {len(filtered_data)/len(data)*100:.1f}%")
        print(f"Saved to: {output_file}")
        
        # Show examples of removed utterances
        if removed_count > 0:
            print(f"\nExamples of removed utterances (< {min_words} words):")
            count = 0
            for entry in data:
                text = entry.get('text', '')
                word_count = len(text.split())
                if word_count < min_words and count < 5:
                    print(f"  {word_count} words: \"{text}\"")
                    count += 1
        
        # Show length distribution of filtered data
        filtered_lengths = [len(entry['text'].split()) for entry in filtered_data]
        if filtered_lengths:
            from collections import Counter
            dist = Counter(filtered_lengths)
            print(f"\nLength distribution of filtered data:")
            for length in sorted(dist.keys())[:10]:  # Show first 10
                count = dist[length]
                percentage = count / len(filtered_data) * 100
                print(f"  {length} words: {count:,} utterances ({percentage:.1f}%)")
            if len(dist.keys()) > 10:
                print("  ...")
    
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    input_file = "combined5.json"
    output_file = "combined5_filtered.json"
    min_words = 3
    
    filter_utterances(input_file, output_file, min_words)

if __name__ == "__main__":
    main()
