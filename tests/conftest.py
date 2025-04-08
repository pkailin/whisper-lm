"""Shared pytest fixtures for configuring the whisper_lm_transformers tests.

This module defines fixtures for various configurations and components such as
language settings, Whisper model configurations, and language model
configurations used in testing the Whisper-LM integration.
"""

import logging
import os
import sys
from unittest.mock import MagicMock

import pytest
import torch
from datasets import Audio, load_dataset
from whisper.audio import load_audio
from whisper.normalizers import BasicTextNormalizer

from whisper_decoder_with_lm import LMOptions

# Include the main repository path
sys.path.insert(0, os.path.abspath(".."))


@pytest.fixture(name="lang")
def fixture_lang():
    """
    Fixture to provide the language code used in the tests.

    Returns:
        str:
            A language code representing the specific language model to be
            tested, e.g., 'eu' for Basque.
    """
    return "eu"


@pytest.fixture(name="whisper_config")
def fixture_whisper_config(lang):  # pylint: disable=
    """
    Provide a configuration dictionary for the Whisper model tests.

    Parameters
    ----------
    lang : str
        Language code provided by the `lang` fixture.

    Returns
    -------
    dict
        Configuration for the Whisper model including keys for 'lang' and
        'model' with the path to the pretrained model.
    """
    return {
        "key": "value",
        "lang": lang,
        "model": f"zuazo-whisper-tiny-{lang}.pt",
    }


@pytest.fixture(name="lm_config")
def fixture_lm_config(lang):
    """
    Provide the configuration for the KenLM model integration in the tests.

    Parameters
    ----------
    lang : str
        Language code used to locate the language-specific KenLM model file.

    Returns
    -------
    dict
        Dictionary containing the path to the KenLM model and tuning
            parameters like alpha and beta.
    """
    path = os.path.dirname(__file__)
    return {
        "path": os.path.join(path, "..", f"5gram-{lang}.bin"),
        "alpha": 0.33582368603855817,
        "beta": 0.6882556478819416,
    }


@pytest.fixture(name="llm_config")
def fixture_llm_config():
    """
    Provide the configuration for the LLM model integration in the tests.

    Returns:
        dict:
            Dictionary containing the path to the LLM model and tuning
            parameters like alpha and beta.
    """
    return {
        "path": "HiTZ/latxa-7b-v1.2",
        "alpha": 2.733293955541733,
        "beta": 0.0017859540619529915,
    }


@pytest.fixture(name="normalizer")
def fixture_normalizer():
    """
    Provide a text normalization function for sentence comparisons.

    Returns:
        function:
            A function from the `BasicTextNormalizer` class configured to
            remove diacritics.
    """
    return BasicTextNormalizer(remove_diacritics=True)


@pytest.fixture(name="example_name", scope="session")
def fixture_example_name():
    """
    Provide a name for selecting a specific audio example.

    Returns
    -------
    str
        The name of the example.
    """
    return "euf_07973_00797482883"


@pytest.fixture(name="audio_path", scope="session")
def fixture_audio_path(example_name):
    """
    Load and provide the path of an audio example.

    Parameters
    ----------
    example_name : str
        The audio example name to use.

    Returns
    -------
    str
        The audio file path.
    """
    audio_path = os.path.join(
        os.path.dirname(__file__),
        "fixtures", f"{example_name}.wav",
    )
    return audio_path


@pytest.fixture(name="audio_array", scope="session")
def fixture_audio_array(audio_path):
    """
    Load and provide the audio array of an audio example.

    Parameters
    ----------
    audio_path : str
        The path of the audio file.

    Returns
    -------
    np.ndarray
        The audio array for the specified example.
    """
    return load_audio(audio_path)


@pytest.fixture(name="text_ref", scope="session")
def fixture_text_ref(example_name):
    """
    Provide the reference transcription for the specified audio example.

    Parameters
    ----------
    example_name : str
        The audio example name to use.

    Returns
    -------
    str
        The reference transcription for the audio example.
    """
    txt_path = os.path.join(
        os.path.dirname(__file__),
        "fixtures", f"{example_name}.txt",
    )
    with open(txt_path, "r", encoding="utf8") as handle:
        text = handle.read().strip()
    return text


# Mock model and dataset to use across tests
@pytest.fixture(name="mock_model")
def fixture_mock_model():
    """Create a Whisper mock model."""
    model = MagicMock()
    model.transcribe.return_value = {"text": "predicted text"}
    model.decode.return_value = [MagicMock(text="decoded text")]
    model.dims.n_mels = 80
    model.device = torch.device("cuda")
    return model


@pytest.fixture(autouse=True)
def reset_lm_options():
    """Fixture to reset LMOptions state before each test."""
    yield
    LMOptions().lm_path = None
    LMOptions().llm_path = None
    LMOptions().lm_alpha = 0.931289039105002
    LMOptions().lm_beta = 1.1834137581510284
    LMOptions().lm_eos = "!?."
    LMOptions().lm_normalize = True
    LMOptions().lm_token_threshold = 4
