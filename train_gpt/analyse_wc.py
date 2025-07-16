import json

def analyze_json_text(file_path):
    """
    Analyze a JSON file containing text entries and calculate statistics
    for entries with more than 2 words.
    """
    
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Filter entries with more than 2 words
    entries_more_than_2_words = []
    
    for entry in data:
        if 'text' in entry:
            words = entry['text'].split()
            if len(words) > 2:
                entries_more_than_2_words.append(entry['text'])
    
    # Calculate statistics
    num_entries = len(entries_more_than_2_words)
    total_words = sum(len(text.split()) for text in entries_more_than_2_words)
    avg_length = total_words / num_entries if num_entries > 0 else 0
    
    # Display results
    print(f"Analysis Results:")
    print(f"================")
    print(f"1. Number of entries with more than 2 words: {num_entries}")
    print(f"2. Total number of words in entries with more than 2 words: {total_words}")
    print(f"3. Average length of entries with more than 2 words: {avg_length:.2f} words")
    
    """
    # Optional: Show the entries that were analyzed
    print(f"\nEntries analyzed:")
    for i, entry in enumerate(entries_more_than_2_words, 1):
        word_count = len(entry.split())
        print(f"{i}. '{entry}' ({word_count} words)")
    """
    return {
        'num_entries': num_entries,
        'total_words': total_words,
        'avg_length': avg_length,
        'entries': entries_more_than_2_words
    }

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.json' with the actual path to your JSON file
    file_path = 'myst_train.json'
    
    try:
        results = analyze_json_text(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
    except Exception as e:
        print(f"An error occurred: {e}")
