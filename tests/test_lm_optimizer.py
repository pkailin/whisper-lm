"""Unit tests for lm_optimizer.py functions and classes.

These tests focus on internal logic and function outputs without actual I/O operations.
"""

from unittest.mock import patch

from lm_optimizer import detect_available_gpus, parse_args


def test_parse_args():
    """Tests the parse_args function by simulating command line input."""
    with patch(
        "sys.argv",
        [
            "lm_optimizer.py",
            "tiny",
            "--lm_path",
            "path/to/lm.bin",
            "--temperature",
            "(0.5)",
        ],
    ):
        args = parse_args()
        assert args.lm_path == "path/to/lm.bin"
        assert args.temperature == (0.5,)


def test_detect_available_gpus():
    """Tests the detect_available_gpus function."""
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}):
        assert detect_available_gpus(None) == ["0", "1"]
    with patch("torch.cuda.device_count", return_value=4):
        assert len(detect_available_gpus(None)) == 4  # Should detect 4 GPUs
