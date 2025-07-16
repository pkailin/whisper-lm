import matplotlib.pyplot as plt
from collections import defaultdict
import os
from jiwer import wer

def read_file(filename):
    """Read and parse the text file into a dictionary."""
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('<DIV>')
            if len(parts) == 3:
                utterance_id = parts[0]
                predicted_text = parts[1]
                label_text = parts[2]
                data[utterance_id] = {
                    'predicted': predicted_text,
                    'label': label_text
                }
    return data

def get_utterance_length(text):
    """Get the length of utterance in words."""
    return len(text.split())

def create_histogram_data(condition_func, file1_data, file2_data, title):
    """Create histogram data based on condition function."""
    length_counts = defaultdict(int)
    
    # Get common utterance IDs
    common_ids = set(file1_data.keys()) & set(file2_data.keys())
    
    for utterance_id in common_ids:
        # Use label text length as the utterance length (since it's the ground truth)
        length = get_utterance_length(file1_data[utterance_id]['label'])
        
        if condition_func(file1_data[utterance_id], file2_data[utterance_id]):
            length_counts[length] += 1

    total_utterances = sum([i for i in length_counts.values()])
    
    # Calculate percentages as percentage of total utterances
    lengths = sorted(length_counts.keys())
    percentages = []
    
    for length in lengths:
        if length < 4:
        #if length < 25: 
            percentage = 0
        else: 
            percentage = (length_counts[length] / total_utterances) * 100
        percentages.append(percentage)
    
    return lengths, percentages, title

def plot_histogram(lengths, percentages, title, filename):
    """Create and save histogram with the specified format."""
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    bars = plt.bar(lengths, percentages, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Utterance Length (words)', fontsize=12)
    plt.ylabel('Percentage of Utterances (%)', fontsize=12)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
   
    max_length = max(lengths)
    # Set x-axis to show reasonable range
    plt.xlim(0, max_length + 1)

    # Set x-axis ticks at intervals of 5
    x_max = min(50, max_length + 2) if max_length > 50 else max_length + 1
    plt.xticks(range(0, int(x_max) + 1, 5))
    
    # Improve layout
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(filename, bbox_inches='tight')
    print(f"Histogram saved as '{filename}'")
    
    plt.close()  # Close the figure to free memory

def main():
    # Get file names from user
    file1_name = input("Enter the name of the first .txt file (After N-gram): ").strip()
    file2_name = input("Enter the name of the second .txt file (Before N-gram): ").strip()
    
    # Check if files exist
    if not os.path.exists(file1_name):
        print(f"Error: File '{file1_name}' not found!")
        return
    if not os.path.exists(file2_name):
        print(f"Error: File '{file2_name}' not found!")
        return
    
    # Read the files
    print("Reading files...")
    file1_data = read_file(file1_name)
    file2_data = read_file(file2_name)
    
    print(f"File 1: {len(file1_data)} utterances")
    print(f"File 2: {len(file2_data)} utterances")
    
    common_ids = set(file1_data.keys()) & set(file2_data.keys())
    print(f"Common utterances: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("Error: No common utterance IDs found between files!")
        return
    
    # Define condition functions for each histogram
    def condition1(data1, data2):
        """Predicted text in file1 != predicted text in file2"""
        return data1['predicted'] != data2['predicted']
    
    def condition2(data1, data2):
        """Predicted text in file1 != predicted text in file2 AND WER of file1 < WER of file2"""
        if data1['predicted'] == data2['predicted']:
            return False
        wer1 = wer(data1['label'], data1['predicted'])
        wer2 = wer(data2['label'], data2['predicted'])
        return wer1 < wer2
    
    def condition3(data1, data2):
        """Predicted text in file1 != predicted text in file2 AND WER of file2 < WER of file1"""
        if data1['predicted'] == data2['predicted']:
            return False
        wer1 = wer(data1['label'], data1['predicted'])
        wer2 = wer(data2['label'], data2['predicted'])
        return wer2 < wer1
    
    # Create histograms
    print("\nGenerating histograms...")
    
    # Histogram 1
    lengths1, percentages1, title1 = create_histogram_data(
        condition1, file1_data, file2_data,
        'Utterances Edited After N-gram Incorporation'
    )
    plot_histogram(lengths1, percentages1, title1, 'prediction_differences.pdf')
    
    # Histogram 2
    lengths2, percentages2, title2 = create_histogram_data(
        condition2, file1_data, file2_data,
        'Utterances with Improved WER After N-gram Incorporation'
    )
    plot_histogram(lengths2, percentages2, title2, 'after_ngram_better_wer.pdf')
    
    # Histogram 3
    lengths3, percentages3, title3 = create_histogram_data(
        condition3, file1_data, file2_data,
        'Utterances with Worsened WER After N-gram Incorporation'
    )
    plot_histogram(lengths3, percentages3, title3, 'before_ngram_better_wer.pdf')
    
    print("\nAll histograms have been generated and saved as PDF files!")
    
    # Print some summary statistics
    print(f"\nSummary Statistics:")
    print(f"Histogram 1 - Average percentage: {sum(percentages1)/len(percentages1):.2f}%" if percentages1 else "No data")
    print(f"Histogram 2 - Average percentage: {sum(percentages2)/len(percentages2):.2f}%" if percentages2 else "No data")
    print(f"Histogram 3 - Average percentage: {sum(percentages3)/len(percentages3):.2f}%" if percentages3 else "No data")

if __name__ == "__main__":
    main()
