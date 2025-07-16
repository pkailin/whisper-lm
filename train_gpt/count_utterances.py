#!/usr/bin/env python3
"""
Analyze the distribution of utterance lengths in combined.json
"""

import json
from collections import Counter

def analyze_length_distribution(json_file):
    """
    Analyze the distribution of utterance lengths.
    
    Args:
        json_file (str): Path to the JSON file
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} utterances from {json_file}")
        
        # Calculate lengths (by number of words)
        word_lengths = []
        char_lengths = []
        
        for entry in data:
            text = entry.get('text', '')
            # Word count (split by whitespace)
            word_count = len(text.split())
            word_lengths.append(word_count)
            
            # Character count
            char_count = len(text)
            char_lengths.append(char_count)
        
        # Count distribution of word lengths
        word_dist = Counter(word_lengths)
        char_dist = Counter(char_lengths)
        
        print("\n" + "="*60)
        print("WORD LENGTH DISTRIBUTION")
        print("="*60)
        print(f"{'Words':<8} {'Count':<8} {'Percentage':<12} {'Bar'}")
        print("-" * 60)
        
        total_utterances = len(word_lengths)
        for length in sorted(word_dist.keys()):
            count = word_dist[length]
            percentage = (count / total_utterances) * 100
            bar = "█" * min(50, int(percentage * 2))  # Scale bar
            print(f"{length:<8} {count:<8} {percentage:<11.2f}% {bar}")
        
        print("\n" + "="*60)
        print("WORD LENGTH STATISTICS")
        print("="*60)
        print(f"Total utterances: {total_utterances:,}")
        print(f"Min word length: {min(word_lengths)}")
        print(f"Max word length: {max(word_lengths)}")
        print(f"Average word length: {sum(word_lengths)/len(word_lengths):.2f}")
        print(f"Median word length: {sorted(word_lengths)[len(word_lengths)//2]}")
        
        # Show most common lengths
        print(f"\nMost common word lengths:")
        for length, count in word_dist.most_common(10):
            print(f"  {length} words: {count:,} utterances ({count/total_utterances*100:.1f}%)")
        
        print("\n" + "="*60)
        print("CHARACTER LENGTH STATISTICS")
        print("="*60)
        print(f"Min character length: {min(char_lengths)}")
        print(f"Max character length: {max(char_lengths)}")
        print(f"Average character length: {sum(char_lengths)/len(char_lengths):.2f}")
        print(f"Median character length: {sorted(char_lengths)[len(char_lengths)//2]}")
        
        # Show some example utterances for different lengths
        print("\n" + "="*60)
        print("EXAMPLE UTTERANCES BY LENGTH")
        print("="*60)
        
        # Group by word length for examples
        length_examples = {}
        for entry in data:
            text = entry.get('text', '')
            word_count = len(text.split())
            if word_count not in length_examples:
                length_examples[word_count] = []
            length_examples[word_count].append(text)
        
        # Show examples for a few different lengths
        example_lengths = [1, 3, 5, 10, 15, 20]
        for length in example_lengths:
            if length in length_examples:
                examples = length_examples[length][:3]  # Show up to 3 examples
                print(f"\n{length} words ({len(length_examples[length])} total):")
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. \"{example}\"")
        
    except FileNotFoundError:
        print(f"Error: File {json_file} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file}: {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    json_file = "combined.json"
    analyze_length_distribution(json_file)

if __name__ == "__main__":
    main()
