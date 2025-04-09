"""Tests the `whisper_evaluate.py script functions.

Note: This scripts needs to be run with `python -m pytest` for the import to work.
"""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from whisper import DecodingResult

from whisper_evaluate import (
    WhisperDataset,
    compute_measures,
    evaluate_with_decode,
    evaluate_with_transcribe,
    get_dtype_and_options,
    int_or_none,
    parse_args,
    parse_none,
    parse_transcribe_options,
    pretty_print_scores,
    save_json,
    set_lm_options,
    str_or_none,
    tuple_type,
)


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("None", None),
        ("some_string", "some_string"),
        ("123", "123"),
    ],
)
def test_parse_none(input_value, expected_output):
    """
    Test the parse_none function to ensure it converts "None" string to None type.

    Parameters:
        input_value (str): The input string to the function.
        expected_output: The expected result after parsing.

    Checks:
        - The function converts "None" exactly to None type.
        - Any other input remains unchanged.
    """
    assert parse_none(input_value) == expected_output


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("None", None),
        ("10", 10),
        ("not_a_number", pytest.raises(ValueError)),
    ],
)
def test_int_or_none(input_value, expected_output):
    """
    Test int_or_none to ensure correct conversion from string to integer or None.

    Parameters:
        input_value (str): A string representation of an integer or 'None'.
        expected_output: The expected integer result or None.

    Checks:
        - Strings correctly converted to integers.
        - "None" is converted to None.
        - Invalid inputs raise a ValueError.
    """
    if expected_output is not None and not isinstance(expected_output, int):
        with expected_output:
            int_or_none(input_value)
    else:
        assert int_or_none(input_value) == expected_output


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("None", None),
        ("text", "text"),
    ],
)
def test_str_or_none(input_value, expected_output):
    """
    Test str_or_none to ensure it correctly returns a string or None.

    Parameters:
        input_value (str): The input string that may represent 'None'.
        expected_output: The expected output after parsing, either a string or None.

    Checks:
        - "None" is converted to None.
        - Any other string is returned as is.
    """
    assert str_or_none(input_value) == expected_output


def test_tuple_type():
    """
    Test tuple_type to ensure correct parsing of tuple strings.

    Checks:
        - String representations of tuples are correctly converted to tuple of floats.
        - Handles single values and ranges.
    """
    input_string = "(0.1, 0.2, 0.3)"
    expected_output = (0.1, 0.2, 0.3)
    assert tuple_type(input_string) == expected_output

    input_string = "0.5"
    expected_output = (0.5,)
    assert tuple_type(input_string) == expected_output

    input_string = "(1.0, 2.0, 3.0, 4.0)"
    expected_output = (1.0, 2.0, 3.0, 4.0)
    assert tuple_type(input_string) == expected_output


@pytest.fixture(name="cli_args")
def fixture_cli_args():
    """
    Provides a list of command-line arguments for use in testing.

    Returns:
        list: A list of strings representing command-line arguments.
    """
    return [
        "model_name_or_path",
        "--language",
        "en",
        "--beam_size",
        "5",
        "--temperature",
        "(0.0)",
        "--lm_path",
        "path/to/lm.bin",
    ]


def test_parse_args(cli_args):
    """
    Test parsing of command line arguments using the parse_args function.

    Tests
    -----
    - Ensure that the model name is correctly parsed.
    - Verify that language settings are correctly parsed as 'en'.
    - Confirm that beam size is correctly parsed as an integer 5.
    - Check that the temperature is correctly parsed into a tuple containing 0.0.
    - Validate that the language model path is correctly parsed.

    Parameters
    ----------
    cli_args : list
        A list of command-line arguments provided by the cli_args fixture.
    """
    with patch("sys.argv", ["whisper_evaluate.py"] + cli_args):
        args = parse_args()
        assert args.model == "model_name_or_path"
        assert args.language == "en"
        assert args.beam_size == 5
        assert args.temperature == (0.0,)
        assert args.lm_path == "path/to/lm.bin"


def test_parse_transcribe_options(cli_args):
    """
    Test parsing of transcription options from command line arguments.

    Tests
    -----
    - Ensure that the transcription task is correctly set to 'transcribe'.
    - Verify that temperature, beam size, and timestamps settings are
      correctly parsed and applied.

    Parameters
    ----------
    cli_args : list
        A list of command-line arguments provided by the cli_args fixture.
    """
    with patch("sys.argv", ["whisper_evaluate.py"] + cli_args):
        args = parse_args()
        transcribe_options = parse_transcribe_options(args)
        assert transcribe_options["task"] == "transcribe"
        assert transcribe_options["temperature"] == (0.0,)
        assert transcribe_options["beam_size"] == 5
        assert transcribe_options["without_timestamps"]


@pytest.fixture(name="mock_args")
def mock_args_fixture():
    """
    Provides a mock argparse.Namespace equivalent to command line arguments.

    Returns:
        argparse.Namespace: Mocked arguments for testing.
    """
    return {
        "lm_path": "path/to/lm.bin",
        "lm_normalize": True,
        "lm_token_threshold": 4,
        "llm_path": "path/to/llm",
        "lm_alpha": 0.5,
        "lm_beta": 0.3,
    }


def test_set_lm_options(mock_args):
    """
    Test set_lm_options to ensure language model options are correctly set.

    Parameters:
        mock_args (dict): Mocked command line arguments parsed as dictionary.

    Checks:
        - Language model settings are correctly applied to the LMOptions singleton.
    """
    set_lm_options(mock_args)
    from whisper_decoder_with_lm import (  # pylint: disable=import-outside-toplevel
        LMOptions,
    )

    assert LMOptions().lm_path == mock_args["lm_path"]
    assert LMOptions().lm_alpha == mock_args["lm_alpha"]
    assert LMOptions().lm_beta == mock_args["lm_beta"]
    assert LMOptions().lm_normalize == mock_args["lm_normalize"]
    assert LMOptions().lm_token_threshold == mock_args["lm_token_threshold"]
    assert LMOptions().llm_path == mock_args["llm_path"]
    assert LMOptions().lm_alpha == mock_args["lm_alpha"]
    assert LMOptions().lm_beta == mock_args["lm_beta"]
    assert LMOptions().lm_normalize == mock_args["lm_normalize"]


def test_compute_measures():
    """
    Test the calculation of ASR performance metrics such as WER and CER.

    This function tests:
        - The correct calculation of WER and CER when there is a typo in the
          predicted text.
        - Checks that the WER and CER values are greater than 0 due to the typo.
    """
    label_texts = ["hello world", "test sentence"]
    predicted_texts = ["hello world", "test sentece"]  # typo in second sentence
    measures = compute_measures(label_texts, predicted_texts)
    assert "wer" in measures
    assert "cer" in measures
    assert measures["wer"] > 0  # Because there's one error in the second sentence
    assert measures["cer"] > 0


@pytest.fixture(name="mock_dataset")
def mock_dataset_fixture():
    """Creates a mock HF dataset value with values."""
    mock_dataset = [{"audio": {"path": "some_path"}, "text": "sample text"}]
    return mock_dataset


def test_evaluate_with_transcribe(mock_model, mock_dataset):
    """
    Test the evaluate_with_transcribe function to ensure it calculates correct measures.
    """
    # Setup
    mock_model.transcribe = MagicMock(return_value={"text": "predicted text"})
    transcribe_options = {"language": "en"}

    # Call the function
    sentence_measures, label_texts, predicted_texts = evaluate_with_transcribe(
        mock_model, mock_dataset, transcribe_options
    )

    # Test checks
    assert len(sentence_measures) > 0
    assert label_texts == ["sample text"]
    assert predicted_texts == ["predicted text"]


def test_get_dtype_and_options(mock_model):
    """
    Test that get_dtype_and_options correctly adjusts dtype and options.
    """
    transcribe_options = {"fp16": True, "temperature": (0)}
    dtype, updated_options = get_dtype_and_options(mock_model, transcribe_options)

    print("Device:", mock_model.device)
    if mock_model.device == torch.device("cuda"):
        assert dtype == torch.float16
        assert updated_options["fp16"] is True
    else:  # FP16 is not supported on CPU
        assert dtype == torch.float32
        assert updated_options["fp16"] is False

    transcribe_options = {"fp16": False, "temperature": (0)}
    dtype, updated_options = get_dtype_and_options(mock_model, transcribe_options)

    assert dtype == torch.float32
    assert updated_options["fp16"] is False


def test_whisper_dataset():
    """Tests the WhisperDataset class."""
    hf_dataset = [{"audio": {"array": np.random.rand(16000)}, "text": "hello world"}]
    dataset = WhisperDataset(hf_dataset, "array", "text", 80, torch.float32, "cpu")
    assert len(dataset) == 1
    mel, text = dataset[0]
    assert isinstance(mel, torch.Tensor)
    assert text == "hello world"


@pytest.fixture(name="mock_audio_data")
def mock_audio_data_fixture():
    """Creates a mock audio tensor representing loaded audio data."""
    # Simulate 10 seconds of audio at 16 kHz
    return torch.rand(16000 * 10)  # pylint: disable=no-member


def test_evaluate_with_decode(mock_model, mock_dataset):
    """
    Test the evaluate_with_decode function to ensure it handles batch decoding properly.
    """
    # Setup the mock DecodingResult
    mock_decoding_result = DecodingResult(
        # Example feature size
        audio_features=torch.randn(80, 3000),  # pylint: disable=no-member
        language="en",
        text="decoded text",
        tokens=[1, 2, 3],  # Example token IDs
        avg_logprob=-0.5,
        no_speech_prob=0.1,
        temperature=0.5,
        compression_ratio=1.5,
    )

    # Setup the mock decode method to return a list of mock DecodingResult
    mock_model.decode = MagicMock(return_value=[mock_decoding_result])
    transcribe_options = {"language": "en", "temperature": (0)}

    # Patch the whisper.load_audio to return a tensor simulating loaded audio
    with patch(
        "whisper.load_audio",
        return_value=torch.rand(16000),  # pylint: disable=no-member
    ):  # Simulating 1 second of audio
        # Call the function
        sentence_measures, label_texts, predicted_texts = evaluate_with_decode(
            mock_model, mock_dataset, transcribe_options, batch_size=1
        )

        # Test checks
        assert len(sentence_measures) > 0
        assert label_texts == ["sample text"]
        assert predicted_texts == ["decoded text"]


def test_pretty_print_scores(capfd):
    """
    Test that pretty_print_scores correctly formats and prints the scores.
    """
    sentence_scores = {"cer": [0.1, 0.2], "wer": [0.3, 0.4]}
    dataset_scores = {"cer": 0.15, "wer": 0.35}
    pretty_print_scores(sentence_scores, dataset_scores)
    out, _ = capfd.readouterr()
    assert "CER" in out
    assert "WER" in out


def test_save_json(tmpdir):
    """
    Test save_json function to ensure it correctly writes data to a JSON file.
    """
    data = {"key": "value"}
    file_path = os.path.join(tmpdir, "test.json")
    save_json(data, file_path)

    # Verify file contents
    with open(file_path, "r", encoding="utf-8") as f:
        contents = json.load(f)
    assert contents == data
