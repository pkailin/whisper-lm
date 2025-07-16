import json
import re
import os
from datasets import load_dataset

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text(text):
    return normalize_processor.tokenizer._normalize(text)

def extract_all_utterances(file_path):
    """
    Extract all character utterances from a text file and return as list of dictionaries.
    """
    utterances = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        dataset = load_dataset("styfeng/TinyDialogues")
        dialogues = dataset['train']['text'] + dataset['validation']['text']
        
        for dialogue in dialogues:
            # Split dialogue into lines and process each line
            lines = dialogue.split('\n')
            
            for line in lines:
                line = line.strip()
                # Look for lines that contain character labels with **
                character_match = re.match(r'\*\*([^*]+)\*\*:', line)
                if character_match:
                    # Extract text between quotes using regex
                    quote_match = re.search(r'"([^"]*)"', line)
                    if quote_match:
                        quoted_text = quote_match.group(1)
                        
                        # Clean up the text - remove \n\n and extra whitespace
                        clean_text = quoted_text.replace('\\n\\n', '').replace('\n\n', '').strip()
                        clean_text = re.sub(r'\s+', ' ', clean_text)
                        
                        if clean_text:
                            # Split by punctuation marks (. ! ?)
                            sentences = re.split(r'[.!?]+', clean_text)
                            
                            for sentence in sentences:
                                sentence = normalize_text(sentence)
                                if sentence:  # Only add non-empty sentences
                                    utterances.append({"text": sentence})
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []
    
    return utterances

def main():
    # File paths
    train_file = "~/rds/rds-altaslp-8YSp2LXTlkY/data/individual_age_data/tinydialogue_age-2_train.txt"
    val_file = "~/rds/rds-altaslp-8YSp2LXTlkY/data/individual_age_data/tinydialogue_age-2_val.txt"
    
    # Expand the ~ to the full path
    train_file = os.path.expanduser(train_file)
    #val_file = os.path.expanduser(val_file)
    
    # Extract utterances from both files
    print("Extracting all character utterances from training file...")
    train_utterances = extract_all_utterances(train_file)
    print(f"Found {len(train_utterances)} utterances in training file.")
    
    #print("Extracting all character utterances from validation file...")
    #val_utterances = extract_all_utterances(val_file)
    #print(f"Found {len(val_utterances)} utterances in validation file.")
    
    # Combine all utterances
    #all_utterances = train_utterances + val_utterances
    all_utterances = train_utterances
    print(f"Total utterances: {len(all_utterances)}")
    
    # Save to JSON file
    output_file = "tinyd_all_full_hf.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_utterances, f, indent=2, ensure_ascii=False)
        print(f"All character utterances saved to {output_file}")
        
        # Display first few examples
        if all_utterances:
            print("\nFirst 5 utterances:")
            for i, utterance in enumerate(all_utterances[:5]):
                print(f"{i+1}. {utterance['text']}")
                
    except Exception as e:
        print(f"Error saving to JSON: {e}")

if __name__ == "__main__":
    main()
