#!/usr/bin/env python
"""Simple script to evaluate Whisper in external audio datasets, not in HF.

The dataset is expected to have the transcriptions in `*.txt` files with the
same name as the audio files.

Example
-------
```bash
$ ./whisper_evaluate_external.py \
    --beam_size 5 \
    --lm_path 5gram-eu.bin \
    --lm_alpha 0.33582368603855817 \
    --lm_beta 0.6882556478819416 \
    ./zuazo-whisper-tiny-eu.pt \
    ./ahomytts
```
"""

import sys
import torch

# GPU Detection - Exit immediately if no GPU is available
def check_gpu_availability():
    """Check if GPU is available and exit if not found."""
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected!")
        print("CUDA is not available. This script requires a GPU to run.")
        print(f"Available devices: {torch.cuda.device_count()}")
        sys.exit(1)
    
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Call GPU check immediately
check_gpu_availability()

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import whisper
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import whisper_decoder_with_lm  # pylint: disable=unused-import # noqa: E501,F401
from whisper_evaluate import (
    compute_measures,
    get_dtype_and_options,
    int_or_none,
    parse_transcribe_options,
    pretty_print_scores,
    save_json,
    set_lm_options,
    str_or_none,
    tuple_type,
)

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import librosa
import random 
import jiwer 

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text(text):
    return normalize_processor.tokenizer._normalize(text)

"""
from whisper_normalizer.english import EnglishTextNormalizer
import re
from num2words import num2words


# Initialize the Whisper text normalizer
normalizer = EnglishTextNormalizer()

def normalize_text(text):
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Apply Whisper's normalizer
    text = normalizer(text)

    # Step 3: Remove all characters except letters, numbers, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)


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

class ASRDataset(Dataset):
    """Simple ASR dataset class to work with generic audio datasets.

    This is designed to work with datasets with wav files including their
    text in txt files with the same name.
    """

    def __init__(self):
        """Create a ASRDataset class instance.

        Parameters
        ----------
        data_dir : str
            The directory containing the dataset.
        """
        #self.data_dir = data_dir
        self.wavdir = "/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/test_myst/wav.scp"
        self.textdir = "/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/test_myst/text"
        self.data = self.load_data()

    def load_data(self):
    #Load audio file paths and their corresponding transcription texts from wav.scp and text files.

    #Returns
    #-------
    #list:
        #A list of tuples, each containing the path to an audio file and its
        #transcription text.
    
        # Load wav paths from wav.scp
        wav_paths = {}
        with open(self.wavdir, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utterance_id, audio_path = parts
                    wav_paths[utterance_id] = audio_path

        # Load transcriptions from text file
        transcriptions = {}
        with open(self.textdir, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utterance_id, text = parts
                    transcriptions[utterance_id] = text

        # Match wavs with their transcriptions
        data = []
        for utterance_id, audio_path in wav_paths.items():
            if utterance_id in transcriptions:
                text = transcriptions[utterance_id]
                data.append((utterance_id, audio_path, text))

        # Comment this out if you want to test all utterances
        #random.seed(42)
        #data = random.sample(data, 7000)

        return data

    def __len__(self):
        """Get the dataset length."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get an audio example from an index."""
        utterance_id, audio_path, text = self.data[idx]
        return utterance_id, audio_path, text


class ASRSpectrogramDataset(Dataset):
    """A PyTorch Dataset class for handling generic ASR datasets.

    It reads .wav files and converting the audio into mel spectrograms using
    Whisper's functions. This class is designed to work with datasets where each
    .wav file has a corresponding .txt file containing its transcription.

    Attributes
    ----------
    data_dir : str
        The directory containing the dataset.
    n_mels : int
        Number of mel frequency bins for the spectrogram.
    dtype : torch.dtype
        The data type to which the audio tensor will be converted.
    device : str
        The device on which to perform computations ('cuda' or 'cpu').

    Parameters
    ----------
    data_dir : str
        The directory that contains the audio files and text transcriptions.
    n_mels : int
        Number of mel bins to use when creating mel spectrograms.
    dtype : torch.dtype
        Data type for the tensors (e.g., torch.float32).
    device : str
        Device for tensor computations (defaults to automatic GPU detection).
    """

    def __init__(self, n_mels=80, dtype=torch.float32, device=None):
        """Initialize the ASRSpectrogramDataset with directory.

        Parameters
        ----------
        data_dir : str
            The directory that contains the audio (.wav) files and their
            corresponding text (.txt) transcriptions.
        n_mels : int, optional
            The number of mel frequency bins to use when converting audio
            signals into mel spectrograms. The default is 80, which is a common
            setting for ASR tasks.
        dtype : torch.dtype, optional
            The data type for the tensors (e.g., torch.float32). It specifies
            the precision with which to perform computations and store the
            tensors. Default is torch.float32, balancing computation speed and
            precision.
        device : str, optional
            The computing device on which the tensor computations are performed.
            Defaults to 'cuda' if available, otherwise uses 'cpu'. Specifying
            the device helps optimize performance by utilizing GPUs.

        Raises
        ------
        ValueError
            If the data directory does not exist or is empty.

        Examples
        --------
        #>>> dataset = ASRSpectrogramDataset("./data/ahomytts", 64, torch.float32)
        #>>> print(dataset[0])  # Retrieves the first audio-text pair
        """
        self.wavdir = "/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/test_myst/wav.scp"
        self.textdir = "/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/test_myst/text"
        #self.data_dir = data_dir
        self.n_mels = n_mels
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.load_data()


    def load_data(self):
    #Load audio file paths and their corresponding transcription texts from wav.scp and text files.

    #Returns
    #-------
    #list:
        #A list of tuples, each containing the path to an audio file and its
        #transcription text.

        # Load wav paths from wav.scp
        wav_paths = {}
        with open(self.wavdir, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utterance_id, audio_path = parts
                    wav_paths[utterance_id] = audio_path

        # Load transcriptions from text file
        transcriptions = {}
        with open(self.textdir, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utterance_id, text = parts
                    transcriptions[utterance_id] = text

        # Match wavs with their transcriptions
        data = []
        for utterance_id, audio_path in wav_paths.items():
            if utterance_id in transcriptions:
                text = transcriptions[utterance_id]
                data.append((audio_path, text))

        return data

    """
    def load_data(self):
        #Load audio file paths and their corresponding transcription texts.

        #Returns
        #-------
        #list:
            #A list of tuples, each containing the path to an audio file and its
            #transcription text.
        
        data = []
        for root, _, files in os.walk(self.data_dir):
            for filename in files:
                if filename.endswith(".wav"):
                    audio_path = os.path.join(root, filename)
                    txt_path = os.path.join(root, Path(audio_path).stem + ".txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as handle:
                            text = handle.read().strip()
                        data.append((audio_path, text))
        return data
    """

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve the mel spectrogram and corresponding text for an index.

        Parameters
        ----------
        idx : int
            The index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - mel : torch.Tensor
                The mel spectrogram tensor.
            - text : str
                The transcription text corresponding to the audio.
        """
        audio_path, text = self.data[idx]
        audio = whisper.load_audio(audio_path)
        if isinstance(audio, list):
            audio = np.array(audio)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)  # pylint: disable=no-member
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio, self.n_mels).to(self.device)
        return mel, text




def parse_args():
    """Parse command line arguments.

    Returns
    -------
    namespace
        The namespace populated with the argument values.
    """
    parser = argparse.ArgumentParser(
        description="Speech-to-Text model evaluation script."
    )
    parser.add_argument(
        "model", help="File of the model to use in OpenAI format."
    )  # noqa: E501
    #parser.add_argument("dataset", help="Path of the dataset.")
    parser.add_argument(
        "--use_decode",
        action="store_true",
        help="Use the `decode()` function with batching support.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="How many samples per batch to load, requires `--use_decode`.",
    )
    parser.add_argument(
        "--normalize-audio",
        "-na",
        action="store_true",
        help="Whether normalize the audio file (not recommended).",
    )
    parser.add_argument(
        "--language",
        "--lang",
        type=str_or_none,
        default=None,
        help="The language in ISO-639-1 (two-letter code).",
    )
    parser.add_argument(
        "--skip_normalize",
        "-n",
        action="store_true",
        help="Whether to normalize the text (enabled by default)",
    )
    parser.add_argument(
        "--with_diacritics",
        action="store_true",
        help="Leave the diacritics when normalizing.",
    )
    parser.add_argument(
        "--temperature",
        type=tuple_type,
        default=(0.0),
        help=(
            "Temperature is a form of controlled randomness. "
            "A list of numbers can be provided separated by commas. "
            "Defaults to 0, which means disabled. The logits will be divided "
            "by this number. "
            "`> 1.0` leads to a more random sampling behaviour. "
            "`< 1.0` makes model more confident in its predictions and "
            "reducing randomness."
        ),
    )
    parser.add_argument(
        "--best_of",
        type=int_or_none,
        default=None,
        help="Number of independent sample trajectories (Beam Search).",
    )
    parser.add_argument(
        "--beam_size",
        type=int_or_none,
        default=5,
        help="Number of beams in beam search, enables Beam Search.",
    )
    parser.add_argument(
        "--patience",
        type=int_or_none,
        default=None,
        help="Patience in beam search.",
    )
    parser.add_argument(
        "--with_timestamps",
        action="store_true",
        help="Enable timestamps prediction.",
    )
    parser.add_argument(
        "--lm_path",
        type=str,
        default=None,
        help="A KenLM n-gram language model path.",
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        default=None,
        help="A Hugging Face language model path or URI.",
    )
    parser.add_argument(
        "--lm_alpha",
        type=float,
        default=None,
        help="KenLM Language Model weight.",
    )
    parser.add_argument(
        "--lm_beta",
        type=float,
        default=None,
        help="KenLM word insertion weight.",
    )
    parser.add_argument(
        "--lm_eos",
        type=str,
        default=None,
        help="KenLM End-of-String characters.",
    )
    parser.add_argument(
        "--lm_normalize",
        type=bool,
        default=True,
        help="Whether to normalize the text for the KenLM.",
    )
    parser.add_argument(
        "--lm_token_threshold",
        type=int,
        default=None,
        help=(
            "Minimum number of tokens in a sequence required before applying "
            "language model scoring. This prevents premature evaluation on "
            "short sequences."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory to save the evaluation outputs. "
            "If not provided, no files will be saved."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output_file",
        required=True, 
    )

    levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    parser.add_argument("--log-level", default="INFO", choices=levels)
    args = parser.parse_args()
    return args


def evaluate_with_transcribe(model, dataset, transcribe_options, output_file, normalizer=None, prompt=0):
    """Evaluate a Whisper ASR model on a given dataset using `transcribe()`.

    Batching is not supported.

    Parameters
    ----------
    model : whisper.Whisper
        The Whisper ASR model to be used for transcription.
    dataset : torch.utils.data.Dataset
        A dataset containing audio data and corresponding ground truth text.
    transcribe_options : dict
        Configuration options for the model's transcribe method, such as
        temperature settings, beam size, etc.
    normalizer : function, optional
        A function used to normalize text data. If None, normalization is
        skipped.

    Returns
    -------
    tuple :
    - sentence_measures : dict
        A dictionary containing lists of computed sentence-level measures
        (e.g., CER, WER) across all examples in the dataset.
    - label_texts : list
        A list of all ground truth texts used for evaluation.
    - predicted_texts : list
        A list of all predicted texts generated by the model.

    Example
    -------
    ```python
    sentence_measures, label_texts, predicted_texts = evaluate_with_transcribe(
        model, dataset, transcribe_options
    )
    ```
    """
    # Aggregating all reference and hypothesis texts for dataset-level metrics
    label_texts = []
    predicted_texts = []

    # Iterate through the dataset
    logging.info("Evaluating the dataset:")
    sentence_measures = defaultdict(list)
    
    with open(output_file, "w", encoding="utf-8") as results_file:
        for utterance_id, audio_path, label_text in tqdm(dataset):
            # Transcribe the audio:
            print('Transcribe Options: ') 
            print(transcribe_options) 

            # Module mapping to full descriptions
            module_descriptions = {
                'ME': 'Magnetism and Electricity',
                'MS': 'Mixtures and Solutions', 
                'VB': 'Variables',
                'WA': 'Water',
                'EE': 'Energy and Electromagnetism',
                'MX': 'Mixtures',
                'SMP': 'Sun, Moon and Planets',
                'SRL': 'Soil, Rocks and Landforms',
                'LS': 'Living Systems'
            }
    
            # Extract module from utterance_id
            # Split by underscore and get the 5th element (index 4)
            parts = utterance_id.split('_')
    
            if len(parts) < 6:
                raise ValueError(f"Invalid utterance_id format: {utterance_id}")
    
            module = parts[4]  # Module is at index 4
    
            # Get the module description
            if module not in module_descriptions:
                raise ValueError(f"Unknown module: {module}")
    
            module_description = module_descriptions[module]
    
            # Create initial prompt
            if prompt == 0: 
                initial_prompt = f"children speech aged 8 to 11 years old about {module_description}"
            elif prompt == 1: 
                initial_prompt = f"children aged 8 to 11 talking about {module_description}"
            elif prompt == 2: 
                initial_prompt = f"children aged 8 to 11 speaking about {module_description}"
            elif prompt == 3: 
                initial_prompt = f"children, 8 to 11 years old, discussing {module_description}"
            elif prompt == 4: 
                initial_prompt = f"children speech: {module_description}, aged 8 to 11"

            print("Prompt: " + initial_prompt)
    
            # Transcribe with the appropriate prompt
            predicted_text = model.transcribe(audio_path, initial_prompt=initial_prompt)["text"]
        
            # Apply normalizations
            label_text = normalize_text(label_text) 
            predicted_text = normalize_text(predicted_text) 

            # Append for dataset-level calculation
            label_texts.append(label_text)
            predicted_texts.append(predicted_text)

            results_file.write(f"{utterance_id}<DIV>{predicted_text}<DIV>{label_text}\n")

            # Compute the sentence-level scores:
            measures = compute_measures(label_text, predicted_text)

            for name, score in measures.items():
                if isinstance(score, (float, int)):
                    sentence_measures[name].append(score)

    return sentence_measures, label_texts, predicted_texts


def evaluate_with_decode(
    model, dataset, transcribe_options, normalizer=None, batch_size=16
):  # pylint: disable=too-many-locals
    """Evaluate a Whisper ASR model on a given dataset using decode().

    This is the only function supporting batching for faster inference.

    Parameters
    ----------
    model : whisper.Whisper
        The Whisper ASR model to be used for decoding.
    dataset : datasets.Dataset
        A dataset containing audio data and corresponding ground truth
        text.
    transcribe_options : dict
        Configuration options for the model's transcribe method, such as
        temperature settings, beam size, etc.
    normalizer : function, optional
        A function used to normalize text data. If None, normalization is
        skipped.
    batch_size : int
        How many samples per batch to load.

    Returns
    -------
    tuple :
    - sentence_measures : dict
        A dictionary containing lists of computed sentence-level measures
        (e.g., CER, WER) across all examples in the dataset.
    - label_texts : list
        A list of all ground truth texts used for evaluation.
    - predicted_texts : list
        A list of all predicted texts generated by the model.

    Example
    -------
    ```python
    sentence_measures, label_texts, predicted_texts = evaluate_with_decode(
        model, dataset, decode_options
    )
    ```
    """
    _, transcribe_options = get_dtype_and_options(model, transcribe_options)
    decode_options = whisper.DecodingOptions(**transcribe_options)

    # Dataset batches loader
    data_loader = DataLoader(dataset, batch_size=batch_size)

    predicted_texts = []
    label_texts = []
    sentence_measures = defaultdict(list)

    for mels, texts in tqdm(data_loader):
        # Decode the whole batch
        results = model.decode(mels, decode_options)

        # Postprocess each example
        for i, result in enumerate(results):
            label_text = texts[i]
            predicted_text = result.text

            # Normalize the text
            if normalizer is not None:
                label_text = normalizer(label_text).strip()
                predicted_text = normalizer(predicted_text).strip()

            # Compute the sentence-level scores:
            measures = compute_measures(label_text, predicted_text)

            for name, score in measures.items():
                if isinstance(score, (float, int)):
                    sentence_measures[name].append(score)

            predicted_texts.append(predicted_text)
            label_texts.append(label_text)

    return sentence_measures, label_texts, predicted_texts


def main():
    """Start the program."""
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    # Print the command line run:
    logging.info("Command: %s", " ".join(sys.argv))

    # Load the STT model:
    logging.info("Loading model: %s", args.model)
    #model = whisper.load_model(args.model)
    model = whisper.load_model(args.model, device="cuda")

    #model = WhisperForConditionalGeneration.from_pretrained("balaji1312/whisper-small-myst-fullfinetune")

    #print(model.keys())           # should include 'dims' and 'model_state_dict'
    #print(model['dims'])          # see model architecture
    #print(list(model['model_state_dict'].keys())[:10])  # list of weight names
    
    # Parse transcription and LM options:
    #transcribe_options = parse_transcribe_options(args)
    transcribe_options = {}
    set_lm_options(args)

    # Load the text normalizer
    normalizer = None

    # Evaluate the sentences:
    if args.use_decode:
        logging.info("Using decode()")
        # Instantiate the ASRDataset
        logging.info("Loading dataset: %s", args.dataset)
        dataset = ASRSpectrogramDataset(data_dir=args.dataset, n_mels=model.dims.n_mels)

        if args.beam_size is not None:
            logging.warning(
                "To use batch_size with beam_size, install a fixed version:"
            )
            logging.warning(
                "- git+https://github.com/zuazo-forks/whisper@v20231117-bsfix"
            )
        sentence_measures, label_texts, predicted_texts = evaluate_with_decode(
            model, dataset, transcribe_options, normalizer, batch_size=args.batch_size
        )
    else:
        logging.info("Using transcribe()")
        # Instantiate the ASRDataset
        logging.info("Loading dataset: ")
        dataset = ASRDataset()
        print("Using Prompt: " + str(args.prompt))

        sentence_measures, label_texts, predicted_texts = evaluate_with_transcribe(
            model, dataset, transcribe_options, args.output_file, normalizer, args.prompt
        )

    # Print sentence-level scores
    print()
    print("Sentence-level scores:")
    for name, score in sentence_measures.items():
        score = np.array(score)
        print(f"Average {name}: {score.mean()} ± {score.std()}")

    # Compute dataset-level scores
    dataset_measures = compute_measures(label_texts, predicted_texts)
    

    try:    
        # Get WER for this combination
        current_wer = dataset_measures.get("wer", float('inf'))

        # Get Detailed Measures
        measures = jiwer.compute_measures(label_texts, predicted_texts)
        total_ref_words = measures['substitutions'] + measures['deletions'] + measures['hits']
        sub_rate = measures['substitutions'] / total_ref_words
        del_rate = measures['deletions'] / total_ref_words
        ins_rate = measures['insertions'] / total_ref_words

        print(f"WER: {current_wer:.4f}, SUB: {sub_rate:.4f}, DEL: {del_rate:.4f}, INS: {ins_rate:.4f}")
    
    except: 
        print(f"Error with WER evaluation!")
    
    
    # Print dataset-level scores
    print()
    print("Dataset-level scores:")
    for name, score in dataset_measures.items():
        if name in ["hypothesis", "ops", "truth"]:
            continue
        print(f"{name}: {score}")

    print()
    print("Summary:")
    pretty_print_scores(sentence_measures, dataset_measures)

    # Saving the output files with the results
    if args.output_dir:
        # Create the output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Save command-line arguments
        save_json(
            vars(args),
            os.path.join(args.output_dir, "args.json"),
        )

        # Save dataset-level metrics
        save_json(
            dataset_measures,
            os.path.join(args.output_dir, "dataset_level_results.json"),
        )

        # Save sentence-level metrics
        measures = []
        for i in range(len(label_texts)):  # pylint: disable=consider-using-enumerate
            result_data = {
                "label_text": label_texts[i],
                "predicted_text": predicted_texts[i],
            }
            for name, scores in sentence_measures.items():
                result_data[name] = scores[i]
            measures.append(result_data)
        save_json(
            measures,
            os.path.join(args.output_dir, "sentence_level_results.json"),
        )
    logging.info("Output directory: %s", args.output_dir)

    logging.info("Finished.")


if __name__ == "__main__":
    main()
