#!/usr/bin/env python3
"""
Script to upload a fine-tuned GPT-2 model to Hugging Face Hub
"""

import os
import json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from huggingface_hub import HfApi, create_repo

def upload_model_to_hf(
    checkpoint_path: str,
    repo_name: str,
    hf_token: str = None,
    private: bool = False,
    model_description: str = None,
    base_model: str = "gpt2"
):
    """
    Upload a fine-tuned GPT-2 model to Hugging Face Hub
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        repo_name: Name for the repository on HF Hub (format: username/model-name)
        hf_token: Hugging Face token (if not provided, will try to use saved token)
        private: Whether to make the repository private
        model_description: Description for the model card
        base_model: Base model that was fine-tuned
    """
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load the model and tokenizer
    try:
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    try:
        # Try to load tokenizer from checkpoint, fallback to base model
        if (checkpoint_path / "tokenizer.json").exists() or (checkpoint_path / "vocab.json").exists():
            tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
        else:
            print(f"No tokenizer found in checkpoint, using base model tokenizer: {base_model}")
            tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return False
    
    # Initialize HF API
    try:
        api = HfApi(token=hf_token)
        print("✓ Hugging Face API initialized")
    except Exception as e:
        print(f"✗ Error initializing HF API: {e}")
        print("Make sure you're logged in with: huggingface-cli login")
        return False
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository created/verified: {repo_name}")
    except Exception as e:
        print(f"✗ Error creating repository: {e}")
        return False
    
    # Create a temporary directory for upload
    temp_upload_dir = Path("./temp_upload")
    temp_upload_dir.mkdir(exist_ok=True)
    
    try:
        # Save model and tokenizer to temp directory
        print("Preparing files for upload...")
        model.save_pretrained(temp_upload_dir)
        tokenizer.save_pretrained(temp_upload_dir)
        
        # Create a model card
        model_card_content = create_model_card(
            repo_name=repo_name,
            base_model=base_model,
            description=model_description,
            checkpoint_path=str(checkpoint_path)
        )
        
        with open(temp_upload_dir / "README.md", "w") as f:
            f.write(model_card_content)
        
        print("✓ Files prepared")
        
        # Upload to Hub
        print(f"Uploading to {repo_name}...")
        api.upload_folder(
            folder_path=temp_upload_dir,
            repo_id=repo_name,
            token=hf_token,
            commit_message=f"Upload fine-tuned model from {checkpoint_path.name}"
        )
        
        print(f"✅ Model successfully uploaded to: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"✗ Error during upload: {e}")
        return False
    
    finally:
        # Clean up temp directory
        import shutil
        if temp_upload_dir.exists():
            shutil.rmtree(temp_upload_dir)
            print("✓ Temporary files cleaned up")

def create_model_card(repo_name: str, base_model: str, description: str = None, checkpoint_path: str = None):
    """Create a model card for the uploaded model"""
    
    default_description = f"Fine-tuned {base_model} model"
    if description:
        default_description = description
    
    model_card = f"""---
language: en
base_model: {base_model}
tags:
- text-generation
- gpt2
- fine-tuned
license: mit
---

# {repo_name.split('/')[-1]}

{default_description}

## Model Details

- **Base Model**: {base_model}
- **Fine-tuned from checkpoint**: {checkpoint_path if checkpoint_path else 'N/A'}
- **Language**: English
- **Model Type**: Causal Language Model

## Usage

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("{repo_name}")
tokenizer = GPT2Tokenizer.from_pretrained("{repo_name}")

# Generate text
input_text = "Your prompt here"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Training Details

This model was fine-tuned using the Hugging Face Transformers library.

## Intended Use

This model is intended for research and educational purposes.

## Limitations

Please be aware that language models can generate biased or inappropriate content. Use responsibly.
"""
    
    return model_card

def main():
    # Configuration - UPDATE THESE VALUES
    CHECKPOINT_PATH = "/home/klp65/rds/hpc-work/whisper-lm/train_gpt/gpt_expanded_corpora/checkpoint-1484745"
    REPO_NAME = "pkailin2002/gpt2-tuned-expanded"  # UPDATE THIS
    MODEL_DESCRIPTION = "Fine-tuned GPT-2 model on speech transcription data"  # UPDATE THIS
    PRIVATE = False  # Set to True if you want a private repository
    
    print("🚀 Starting model upload to Hugging Face Hub")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Repository: {REPO_NAME}")
    print("-" * 50)
    
    # Check if user needs to update configuration
    if "your-username" in REPO_NAME:
        print("⚠️  Please update the REPO_NAME variable with your actual username and desired model name")
        print("   Format: 'your-hf-username/your-model-name'")
        return
    
    success = upload_model_to_hf(
        checkpoint_path=CHECKPOINT_PATH,
        repo_name=REPO_NAME,
        private=PRIVATE,
        model_description=MODEL_DESCRIPTION
    )
    
    if success:
        print("\n🎉 Upload completed successfully!")
        print(f"Your model is now available at: https://huggingface.co/{REPO_NAME}")
    else:
        print("\n❌ Upload failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
