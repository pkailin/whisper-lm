import os
import glob
import random
import argparse
from tqdm import tqdm

def read_trn_file(filepath):
    """Read content from a .trn transcription file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Read and return the text content, stripping whitespace
            return file.read().strip()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def process_directory(base_dir, recursive=True):
    """Process all .trn files in a directory and its subdirectories."""
    if recursive:
        pattern = os.path.join(base_dir, "**", "*.trn")
        trn_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(base_dir, "*.trn")
        trn_files = glob.glob(pattern)
    
    all_texts = []
    
    print(f"Processing {len(trn_files)} .trn files from {base_dir}...")
    for filepath in tqdm(trn_files):
        text = read_trn_file(filepath)
        if text:  # Only add non-empty texts
            all_texts.append(text)
            
    return all_texts

def save_to_file(texts, output_file):
    """Save a list of texts to a file, one text per line."""
    print(f"Saving {len(texts)} texts to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text + "\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from .trn transcription files")
    parser.add_argument('--train_dir', type=str, default='/home/klp65/rds/hpc-work/myst_child_conv_speech/data/train',
                        help='Directory containing training data')
    parser.add_argument('--dev_dir', type=str, default='/home/klp65/rds/hpc-work/myst_child_conv_speech/data/development',
                        help='Directory containing development data')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for the processed datasets')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Portion of training data to use for validation if no dev set provided')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Process training data
    train_texts = process_directory(args.train_dir)
    print(f"Found {len(train_texts)} training examples")
    
    # Process development data if available
    if os.path.exists(args.dev_dir):
        dev_texts = process_directory(args.dev_dir)
        print(f"Found {len(dev_texts)} development examples")
    else:
        # Split training data if no development set is provided
        print(f"No development directory found. Splitting training data with validation size {args.val_size}")
        random.shuffle(train_texts)
        split_idx = int(len(train_texts) * (1 - args.val_size))
        dev_texts = train_texts[split_idx:]
        train_texts = train_texts[:split_idx]
        print(f"Split into {len(train_texts)} training and {len(dev_texts)} validation examples")
    
    # Save the datasets
    train_output = os.path.join(args.output_dir, "train.txt")
    dev_output = os.path.join(args.output_dir, "validation.txt")
    
    save_to_file(train_texts, train_output)
    save_to_file(dev_texts, dev_output)
    
    print(f"Dataset preparation complete. Files saved to {args.output_dir}")
    print(f"Training file: {train_output}")
    print(f"Validation file: {dev_output}")
    
    # Print information about how to use these files with the HuggingFace script
    print("\nTo use these files with the HuggingFace training script, include the following arguments:")
    print(f"--train_file {train_output} --validation_file {dev_output}")

if __name__ == "__main__":
    main()
