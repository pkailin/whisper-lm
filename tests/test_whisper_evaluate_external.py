"""Tests the `whisper_evaluate.py script functions.

Note: This scripts needs to be run with `python -m pytest` for the import to work.
"""

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
import torch

from whisper_evaluate_external import (
    ASRDataset,
    ASRSpectrogramDataset,
    evaluate_with_decode,
    evaluate_with_transcribe,
    main,
    parse_args,
)


@pytest.fixture(name="mock_audio_files")
def fixture_mock_audio_files(tmp_path):
    """
    Creates a mock directory with audio files and corresponding text files.
    """
    (tmp_path / "audio1.wav").write_text("Sample audio content", encoding="utf-8")
    (tmp_path / "audio1.txt").write_text(
        "This is a sample transcription.", encoding="utf-8"
    )
    return tmp_path


def test_asr_dataset(mock_audio_files):
    """
    Test the ASRDataset class to ensure it correctly loads audio and text data.
    """
    dataset = ASRDataset(data_dir=str(mock_audio_files))
    assert len(dataset) == 1
    audio_path, text = dataset[0]
    assert audio_path.endswith("audio1.wav")
    assert text == "This is a sample transcription."


@pytest.fixture(name="mock_audio_tensor")
def mock_audio_tensor_fixture():
    """Returns a tensor simulating audio data."""
    # Simulating 1 second of random audio data
    return torch.randn(1, 16000)  # pylint: disable=no-member


def test_asrspectrogram_dataset(mock_audio_files, mock_audio_tensor):
    """
    Test the ASRSpectrogramDataset class to ensure it correctly processes specs.
    """
    with patch("whisper.load_audio", return_value=mock_audio_tensor):
        dataset = ASRSpectrogramDataset(data_dir=str(mock_audio_files))
        mel, text = dataset[0]
        assert isinstance(mel, torch.Tensor)
        assert text == "This is a sample transcription."


@pytest.fixture(name="cli_args")
def fixture_cli_args():
    """
    Provides a list of command-line arguments for use in testing.

    Returns:
        list: A list of strings representing command-line arguments.
    """
    return [
        "model_name_or_path",
        "./ahomytts",
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
    """
    with patch("sys.argv", ["whisper_evaluate_external.py"] + cli_args):
        args = parse_args()
        assert args.model == "model_name_or_path"
        assert args.dataset == "./ahomytts"
        assert args.beam_size == 5


def test_evaluate_with_transcribe(mock_model, mock_audio_files):
    """
    Test evaluate_with_transcribe to ensure it correctly computes measures.
    """
    dataset = ASRDataset(data_dir=str(mock_audio_files))
    transcribe_options = {"language": "en", "temperature": 0.0}
    with patch(
        "whisper.load_audio",
        return_value=torch.rand(16000),  # pylint: disable=no-member
    ):
        sentence_measures, label_texts, predicted_texts = evaluate_with_transcribe(
            mock_model, dataset, transcribe_options
        )
    assert len(sentence_measures) > 0
    assert len(label_texts) == len(predicted_texts) == 1
    assert predicted_texts[0] == "predicted text"


def test_evaluate_with_decode(mock_model, mock_audio_files):
    """
    Test evaluate_with_decode to ensure it handles batch decoding properly.
    """
    dataset = ASRSpectrogramDataset(data_dir=str(mock_audio_files))
    transcribe_options = {"language": "en", "temperature": 0.0}
    with patch(
        "whisper.load_audio",
        return_value=torch.rand(16000),  # pylint: disable=no-member
    ):
        sentence_measures, label_texts, predicted_texts = evaluate_with_decode(
            mock_model, dataset, transcribe_options
        )

    assert len(sentence_measures) > 0
    assert len(label_texts) == len(predicted_texts)


def test_main_integration(cli_args):
    """
    Integration test for the main function to ensure the correct sequence of operations.
    """
    with patch("sys.argv", ["script_name"] + cli_args), patch(
        "whisper.load_model"
    ) as mock_load_model, patch(
        "whisper_evaluate_external.ASRDataset"
    ) as mock_asr_dataset, patch(
        "whisper_evaluate_external.evaluate_with_transcribe"
    ) as mock_evaluate, patch(
        "builtins.print"
    ) as mock_print:
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"text": "predicted text"}
        mock_model.decode.return_value = [MagicMock(text="decoded text")]
        mock_model.dims.n_mels = 80
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_model.device = torch.device(device)
        mock_asr_dataset.return_value = [("/path/to/audio.wav", "text")]
        mock_evaluate.return_value = (defaultdict(list), ["text"], ["predicted text"])

        main()

        mock_load_model.assert_called_once()
        mock_asr_dataset.assert_called_once()
        mock_evaluate.assert_called_once()
        # Check if print was called, indicating output was handled
        mock_print.assert_called()
