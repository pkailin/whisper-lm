#!/usr/bin/env python3
"""
Analyze the distribution of utterance lengths in combined.json and save histogram
"""

import json
import matplotlib.pyplot as plt
from collections import Counter

def analyze_length_distribution(json_file):
    """
    Analyze the distribution of utterance lengths and create histogram.

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

        # Create histogram
        create_histogram(word_dist, len(word_lengths))

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

def create_histogram(word_dist, total_utterances):
    """
    Create and save histogram of word length distribution.
    
    Args:
        word_dist (Counter): Counter object with word length distribution
        total_utterances (int): Total number of utterances
    """
    # Prepare data for histogram
    lengths = sorted(word_dist.keys())
    percentages = [(word_dist[length] / total_utterances) * 100 for length in lengths]
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    bars = plt.bar(lengths, percentages, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title('Distribution of Utterance Lengths (After Filtering)', fontsize=16, pad=20)
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Percentage of Utterances (%)', fontsize=12)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    """
    # Add value labels on top of bars for significant percentages
    for bar, length, percentage in zip(bars, lengths, percentages):
        if percentage >= 1.0:  # Only label bars with >= 1% to avoid clutter
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    """

    # Set x-axis to show reasonable range
    max_length = max(lengths)
    if max_length > 50:
        plt.xlim(0, min(50, max_length + 2))  # Limit x-axis if too many long utterances
        plt.text(0.98, 0.98, f'Note: Some utterances extend to {max_length} words',
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    else:
        plt.xlim(0, max_length + 1)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the histogram
    output_filename = 'word_length_histogram_filtered.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved as '{output_filename}'")
    
    # Also save as PDF for better quality
    pdf_filename = 'word_length_histogram.pdf'
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"Histogram also saved as '{pdf_filename}'")
    
    # Show the plot (optional - comment out if running in headless environment)
    # plt.show()
    
    plt.close()  # Close the figure to free memory

def main():
    json_file = "combined4_filtered.json"
    analyze_length_distribution(json_file)

if __name__ == "__main__":
    main()
