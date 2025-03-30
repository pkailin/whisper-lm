"""Tests for the Whisper-LM hack in the `whisper_decoder_with_lm.py` file.

This file tests the integration of the language models with the Whisper
generation process works correctly, focusing on output correctness.

Note: This scripts needs to be run with `python -m pytest` for the import to work.
"""

import os

import pytest
import whisper

# Hack Whisper to support LM and load the options interface to set it up:
from whisper_decoder_with_lm import LMOptions


@pytest.mark.usefixtures(
    "whisper_config",
    "lm_config",
    "audio_path",
    "text_ref",
    "normalizer",
)
def test_whisper_with_lm(whisper_config, lm_config, audio_path, text_ref, normalizer):
    """Test the Whisper-LM hack for correct text generation with LMs.

    Parameters
    ----------
    whisper_config:
        Configuration fixture for Whisper model.
    lm_config:
        Language model configuration fixture.
    audio_array:
        Audio data for testing.
    text_ref:
        Reference text for comparison.
    normalizer:
        Text normalizing function.
    """
    # Hack Whisper to support LM and load the options interface to set it up:

    # Set original Whisper transcription options (this is important):
    decode_options = {
        "language": whisper_config["lang"],
        "without_timestamps": True,
        "temperature": 0.0,
        "beam_size": 5,
    }
    transcribe_options = {"task": "transcribe", **decode_options}

    # Set LM-specific options:
    LMOptions().lm_path = lm_config["path"]
    LMOptions().lm_alpha = lm_config["alpha"]
    LMOptions().lm_beta = lm_config["beta"]

    # Load the model and transcribe the audio:
    model = whisper.load_model("zuazo-whisper-tiny-eu.pt")
    result = model.transcribe(audio_path, **transcribe_options)

    text_ref = normalizer(text_ref).strip()
    text_out = normalizer(result["text"]).strip()

    assert text_out == text_ref


@pytest.mark.skipif(not os.getenv("TEST_LLM"), reason="Skipping LLM tests.")
@pytest.mark.usefixtures(
    "whisper_config", "lm_config", "audio_path", "text_ref", "normalizer"
)
def test_whisper_with_llm(whisper_config, llm_config, audio_path, text_ref, normalizer):
    """Test the Whisper-LLM hack for correct text generation with LMs.

    Parameters
    ----------
    whisper_config:
        Configuration fixture for Whisper model.
    lm_config:
        Language model configuration fixture.
    audio_array:
        Audio data for testing.
    text_ref:
        Reference text for comparison.
    normalizer:
        Text normalizing function.
    """

    # Set original Whisper transcription options (this is important):
    decode_options = {
        "language": whisper_config["lang"],
        "without_timestamps": True,
        "temperature": 0.0,
        "beam_size": 5,
    }
    transcribe_options = {"task": "transcribe", **decode_options}

    # Set LM-specific options:
    LMOptions().lm_path = llm_config["path"]
    LMOptions().lm_alpha = llm_config["alpha"]
    LMOptions().lm_beta = llm_config["beta"]

    # Load the model and transcribe the audio:
    model = whisper.load_model("zuazo-whisper-tiny-eu.pt")
    result = model.transcribe(audio_path, **transcribe_options)

    text_ref = normalizer(text_ref).strip()
    text_out = normalizer(result["text"]).strip()

    assert text_out == text_ref
