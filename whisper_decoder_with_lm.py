"""Extends the internal Whisper classes to support a KenLM.

This code is still used here, but has been recently moved to the following
whisper fork: https://github.com/zuazo-forks/whisper/tree/lm-simple

Example
-------
Download and convert the model to OpenAI format:

```shell
# Converts the model from Hugging Face to OpenAI format:
$ ./convert_hf_to_openai.py \
    --checkpoint zuazo/whisper-medium-eu \
    --whisper_dump_path zuazo-whisper-medium-eu.pt
```

Transcription example:

```python
>>> # Converts the model from Hugging Face to OpenAI format:
>>> from convert_hf_to_openai import convert_tfms_to_openai_whisper
>>> convert_tfms_to_openai_whisper(
...   "zuazo/whisper-medium-eu", "zuazo-whisper-medium-eu.pt"
... )
HF model path: zuazo/whisper-medium-eu
OpenAI model path: zuazo-whisper-medium-eu.pt

>>> # Hack Whisper to support LM and load the options interface to set it up:
>>> from whisper_decoder_with_lm import LMOptions

>>> # Select an audio file:
>>> audio_path = "tests/fixtures/common_voice_eu_18591439.mp3"

>>> # Set original Whisper transcription options:
>>> decode_options = {
...     "language": "eu",
...     "without_timestamps": True,
...     "temperature": 0.0,  # this is important
...     "beam_size": 5,
...     "patience": None,
... }
>>> transcribe_options = {"task": "transcribe", **decode_options}

>>> # Set LM-specific options:
>>> LMOptions().lm_path = "5gram-eu.bin"
>>> LMOptions().lm_alpha = 0.33582368603855817
>>> LMOptions().lm_beta = 0.6882556478819416

>>> # Load the model and transcribe the audio:
>>> import whisper
>>> model = whisper.load_model("zuazo-whisper-medium-eu.pt")
>>> result = model.transcribe(audio_path, **transcribe_options)
>>> result["text"]
'Non demontre dago langraizoka eta non bolikosta?'

```
"""

import logging
import string
from threading import Lock
from typing import Optional, Tuple

import kenlm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from whisper import Whisper
from whisper.decoding import BeamSearchDecoder, DecodingOptions, DecodingTask, Inference
from whisper.normalizers import BasicTextNormalizer
from whisper.tokenizer import Tokenizer

# Extending the DecodingOptions class to support an LM
# ====================================================


class LMOptions:  # pylint: disable=too-few-public-methods
    """Singleton class to pass the LM options to the Beam Search algorithm.

    I did not found a better way to pass the configuration options to the
    `BeamSearchDecoderWithLM` class.
    """

    _instance = None

    # A KenLM n-gram language model path:
    lm_path: str = None

    # Hugging Face LM model path or URI:
    llm_path: str = None

    # The maximum of the alpha hyperparameter of the CTC decoder explored
    # during hyperparameter optimization. Language Model weight.
    lm_alpha: float = 0.931289039105002

    # End of string character list for the LM:
    lm_eos: str = "!?."

    # The maximum beta hyperparameter of the CTC decoder explored during
    # hyperparameter optimization. Word insertion weight.
    lm_beta: float = 1.1834137581510284

    # Whether to normalize text before sending it to the languge model:
    lm_normalize: bool = True

    # Minimum number of tokens in a sequence required before applying language
    # model scoring. This prevents premature evaluation on short sequences.
    lm_token_threshold: int = 4

    def __new__(cls):
        """
        Create or return the LMOptions instance.

        This method implements the singleton pattern which ensures that only
        one instance of the LMOptions class exists.

        Returns
        -------
        LMOptions
            The single instance of LMOptions.

        Example
        -------
        >>> options1 = LMOptions()
        >>> LMOptions().lm_path = "5gram-eu.bin"
        >>> options2 = LMOptions()
        >>> options1 is options2
        True
        """
        if not cls._instance:
            cls._instance = super(LMOptions, cls).__new__(cls)
        return cls._instance


# New Beam Search class with LM support (KenLM)
# =============================================


class BeamSearchDecoderWithLM(
    BeamSearchDecoder
):  # pylint: disable=too-many-instance-attributes
    """New Beam Search class with LM support (KenLM)."""

    def __init__(
        self,
        beam_size: int,
        tokenizer: Tokenizer,
        inference: Inference,
        patience: Optional[float] = None,
        lm_path: Optional[str] = None,
        lm_alpha: Optional[float] = None,
        lm_beta: Optional[float] = None,
        lm_eos: Optional[str] = None,
        lm_normalize: Optional[bool] = True,
    ):  # pylint: disable=too-many-arguments
        """
        Initialize the beam search decoder with n-gram language model support.

        Parameters
        ----------
        beam_size : int
            The number of beams to use in the search process.
        tokenizer : Tokenizer
            The tokenizer instance used for tokenizing input text and
            detokenizing output tokens.
        inference : Inference
            The inference model used to predict the next token based on the
            current state.
        patience : Optional[float], default=None
            The patience parameter controls how long the search should wait for
            a better candidate before terminating the search early.
        lm_path : Optional[str], default=None
            The file path to the pre-trained KenLM language model.
        lm_alpha : Optional[float], default=None
            The weight (alpha) of the language model score.
        lm_beta : Optional[float], default=None
            The weight (beta) applied to the word count within the language
            model scoring.
        lm_eos : Optional[str], default=None
            Characters considered as end-of-sentence markers.
        lm_normalize : Optional[bool], default=True
            Indicates whether to normalize the text before scoring with the
            language model.
        """
        super().__init__(beam_size, tokenizer.eot, inference, patience)
        self.tokenizer = tokenizer
        self.special_tokens = list(self.tokenizer.special_tokens.values())
        self.lm_model = (
            kenlm.Model(lm_path) if lm_path is not None else None
        )  # pylint: disable=c-extension-no-member
        self.lm_alpha = lm_alpha or 0.0
        self.lm_beta = lm_beta or 0.0
        self.lm_eos = lm_eos or ""  # end of sentence chars
        self.lm_eow = set(string.punctuation)  # end of word chars
        self.lm_normalize = lm_normalize  # whether to normalize the LM text
        self.lm_normalizer = BasicTextNormalizer()  # normalize for the KenLM
        self.finished_sequences = None

    def lm_score_and_word_count(self, sequence) -> Tuple[float, int]:
        """Get n-gram language model score and word count for a sequence.

        Parameters
        ----------
        sequence : tuple of int
            A sequence of token IDs.

        Returns
        -------
        float
            The language model score for the decoded text of the sequence.
        int
            The number of words in the decoded text of the sequence.
        """
        if not self.lm_model:
            return None, 0.0

        # Convert sequence of tokens to text
        sequence = tuple(t for t in sequence if t not in self.special_tokens)
        if len(sequence) < LMOptions().lm_token_threshold:
            return None, 0.0
        text = self.tokenizer.decode(sequence)

        # Early return for empty text
        if not text:
            return None, 0.0
        logging.debug('LM text: "%s"', text)

        # Normalize the text
        if self.lm_normalize:
            normalized_text = self.lm_normalizer(text)
        else:
            normalized_text = text
        logging.debug('LM text normalized: "%s"', normalized_text)

        # Check for end of sentence and end of word:
        eos = text[-1] in self.lm_eos

        word_count = len(normalized_text.split())
        logging.debug("Word count: %d", word_count)

        # In KenLM, the most probable sequences have a higher score:
        score = self.lm_model.score(normalized_text, bos=True, eos=eos)
        logging.debug("LM score: %f", score)

        return score, word_count

    def update(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements # noqa: E501
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Update the beam search state with language model scoring.

        This method performs a beam search step and updates internal states,
        such as finished sequences and token caches. The beam search step
        includes LM scoring for ranking beam candidates.

        The method internally:

        1. Calculates the cumulative log probabilities for potential beam
           candidates by considering both the model's predictions and optional
           LM scores.
        2. Ranks the candidates and keeps the top 'beam_size' sequences for
           each audio sample.
        3. Checks and keeps track of sequences that have finished decoding.

        This code is based on `BeamSearchDecoder.update()`, but with the
        additional integration of language model scoring.

        Parameters
        ----------
        tokens : Tensor)
            Current tokens in the beam. Should have shape
            [n_audio * beam_size, seq_len], where n_audio is the number of
            audio samples and beam_size is the number of beams.
        logits : Tensor
            Raw prediction scores for the next token, of shape
            [n_audio * beam_size, vocab_size].
        sum_logprobs : Tensor
            Cumulative log probabilities of the sequences in the beam so far.
            Should have shape [n_audio * beam_size].

        Returns
        -------
        Tuple[Tensor, bool]:
            - A tensor with the updated tokens for each beam, of shape
              [n_audio * beam_size, seq_len].
            - A boolean indicating if the beam search is completed for all
              audio samples.

        Raises
        ------
        ValueError:
            If the tokens tensor's shape is not divisible by the beam size.
        """
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible
            # candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(
                    *logprobs[idx].topk(self.beam_size + 1)
                ):  # noqa: E501
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    logging.debug("AC score (new_logprob): %f", new_logprob)
                    sequence = tuple(prefix + [token.item()])
                    # Adjust the score by adding the LM score:
                    lm_score, wordc = self.lm_score_and_word_count(sequence)
                    if lm_score is not None:  # if it is a word boundary
                        lm_adjusted_score = (
                            new_logprob
                            + self.lm_alpha * lm_score
                            + wordc * self.lm_beta
                        )
                        scores[sequence] = lm_adjusted_score
                    else:
                        scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences
            # for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(  # pylint: disable=no-member
            next_tokens, device=tokens.device
        )  # pylint: disable=no-member
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(
                newly_finished, key=newly_finished.get, reverse=True
            ):  # noqa: E501
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed


class LLMSingleton:
    """
    Handle LLM class loading in GPU memory.

    A singleton class to manage the loading and caching of language models and
    tokenizers to ensure that each model and tokenizer is instantiated only
    once throughout the application.

    Attributes
    ----------
    _models : dict
        A dictionary to store model instances indexed by model names.
    _tokenizers : dict
        A dictionary to store tokenizer instances indexed by tokenizer names.
    _models_lock : Lock
        A threading lock to ensure thread-safe access to the `_models` dictionary.
    _tokenizers_lock : Lock
        A threading lock to ensure thread-safe access to the `_tokenizers` dictionary.

    Methods
    -------
    get_model(model_name)
        Retrieves a model instance for the given model name or loads it if not
        already present.
    get_tokenizer(tokenizer_name)
        Retrieves a tokenizer instance for the given tokenizer name or loads it
        if not already present.
    """

    _models = {}
    _tokenizers = {}
    _models_lock = Lock()
    _tokenizers_lock = Lock()

    @classmethod
    def get_model(cls, model_name):
        """
        Retrieve or load a model by name ensuring singleton instantiation.

        Parameters
        ----------
        model_name : str
            The identifier name of the model to be loaded or retrieved.

        Returns
        -------
        model : PreTrainedModel
            An instance of `AutoModelForCausalLM` corresponding to the specified
            `model_name`.

        Notes
        -----
        If the model is not already loaded, it will fetch the model from
        HuggingFace's repository using the `AutoModelForCausalLM.from_pretrained`
        method, cache it, and return the instance. If already loaded, it simply
        returns the cached instance.
        """
        with cls._models_lock:
            if model_name not in cls._models:
                logging.debug("Loading model: %s", model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                cls._models[model_name] = model
            return cls._models[model_name]

    @classmethod
    def get_tokenizer(cls, tokenizer_name):
        """
        Retrieve or load a tokenizer by name ensuring singleton instantiation.

        Parameters
        ----------
        tokenizer_name : str
            The identifier name of the tokenizer to be loaded or retrieved.

        Returns
        -------
        tokenizer : PreTrainedTokenizer
            An instance of `AutoTokenizer` corresponding to the specified
            `tokenizer_name`.

        Notes
        -----
        If the tokenizer is not already loaded, it will fetch the tokenizer
        from HuggingFace's repository using the `AutoTokenizer.from_pretrained`
        method, cache it, and return the instance. If already loaded, it simply
        returns the cached instance.
        """
        with cls._tokenizers_lock:
            if tokenizer_name not in cls._tokenizers:
                logging.debug("Loading tokenizer: %s", tokenizer_name)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                cls._tokenizers[tokenizer_name] = tokenizer
            return cls._tokenizers[tokenizer_name]


class BeamSearchDecoderWithLLM(BeamSearchDecoderWithLM):
    """Beam Search class with support for Llama (Hugging Face LLM)."""

    def __init__(
        self,
        beam_size: int,
        tokenizer: Tokenizer,
        inference: Inference,
        patience: Optional[float] = None,
        llm_path: Optional[str] = None,
        lm_alpha: Optional[float] = None,
        lm_beta: Optional[float] = None,
        lm_eos: Optional[str] = None,
        lm_normalize: Optional[bool] = True,
    ):  # pylint: disable=too-many-arguments
        """
        Initialize the beam search decoder with large language model support.

        Parameters
        ----------
        beam_size : int
            The number of beams to use in the search process.
        tokenizer : Tokenizer
            The tokenizer instance used for tokenizing input text and
            detokenizing output tokens.
        inference : Inference
            The inference model used to predict the next token based on the
            current state.
        patience : Optional[float], default=None
            The patience parameter controls how long the search should wait for
            a better candidate before terminating the search early.
        llm_path : Optional[str], default=None
            The HF name or path to the pre-trained LLM.
        lm_alpha : Optional[float], default=None
            The weight (alpha) of the language model score.
        lm_beta : Optional[float], default=None
            The weight (beta) applied to the word count within the language
            model scoring.
        lm_eos : Optional[str], default=None
            Characters considered as end-of-sentence markers.
        lm_normalize : Optional[bool], default=True
            Indicates whether to normalize the text before scoring with the
            language model.
        """
        super().__init__(
            beam_size,
            tokenizer,
            inference,
            patience,
            None,
            lm_alpha,
            lm_beta,
            lm_eos,
            lm_normalize,
        )

        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the models, only once
        if llm_path:
            self.llm_model = LLMSingleton.get_model(llm_path).to(self.device)
            self.llm_tokenizer = LLMSingleton.get_tokenizer(llm_path)
        else:
            self.llm_model = self.llm_tokenizer = None

    def lm_score_and_word_count(self, sequence) -> Tuple[float, int]:
        """Get large language model score and word count for a sequence.

        Parameters
        ----------
        sequence : tuple of int
            A sequence of token IDs.

        Returns
        -------
        float
            The language model score for the decoded text of the sequence.
        int
            The number of words in the decoded text of the sequence.
        """
        # Similar implementation for LLM
        # Convert sequence of tokens to text
        sequence = tuple(t for t in sequence if t not in self.special_tokens)
        if len(sequence) < LMOptions().lm_token_threshold:
            return None, 0.0
        text = self.tokenizer.decode(sequence)

        # Early return for empty text
        if not text:
            return None, 0.0
        logging.debug('LLM text: "%s"', text)

        # Normalize the text
        if self.lm_normalize:
            normalized_text = self.lm_normalizer(text)
        else:
            normalized_text = text
        logging.debug('LLM text normalized: "%s"', normalized_text)

        word_count = len(normalized_text.split())
        logging.debug("Word count: %d", word_count)

        # Tokenize the input
        tokens = self.llm_tokenizer(normalized_text, return_tensors="pt").to(
            self.device
        )

        # Get input IDs and attention mask
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # outputs = self.llm_model(**tokens)
        # Calculate output from the model
        outputs = self.llm_model(
            input_ids, attention_mask=attention_mask, labels=input_ids
        )

        # Get the log probabilities of the last token
        log_probs = outputs.logits[:, -1, :].softmax(dim=-1)
        # Use the highest log probability as the score
        max_log_prob = log_probs.max().item()
        # Convert from natural log to log10 (like KenLM)
        score = max_log_prob  # / math.log(10) * -100

        logging.debug("LLM score: %f", score)

        return score, word_count


class BeamSearchDecoderWithLMAndLLM(BeamSearchDecoderWithLM):
    """Beam Search class with support for KenLM and Hugging Face LLM together.

    It uses the word count weight (the beta) as the large language weight.
    """

    def __init__(
        self,
        beam_size: int,
        tokenizer: Tokenizer,
        inference: Inference,
        patience: Optional[float] = None,
        lm_path: Optional[str] = None,
        llm_path: Optional[str] = None,
        lm_alpha: Optional[float] = None,
        lm_beta: Optional[float] = None,
        lm_eos: Optional[str] = None,
        lm_normalize: Optional[bool] = True,
    ):  # pylint: disable=too-many-arguments
        """
        Initialize the beam search decoder with n-gram and large LMs.

        Parameters
        ----------
        beam_size : int
            The number of beams to use in the search process.
        tokenizer : Tokenizer
            The tokenizer instance used for tokenizing input text and
            detokenizing output tokens.
        inference : Inference
            The inference model used to predict the next token based on the
            current state.
        patience : Optional[float], default=None
            The patience parameter controls how long the search should wait for
            a better candidate before terminating the search early.
        lm_path : Optional[str], default=None
            The file path to the pre-trained KenLM language model.
        llm_path : Optional[str], default=None
            The HF name or path to the pre-trained LLM.
        lm_alpha : Optional[float], default=None
            The weight (alpha) of the language model score.
        lm_beta : Optional[float], default=None
            The weight (beta) applied to the word count within the language
            model scoring.
        lm_eos : Optional[str], default=None
            Characters considered as end-of-sentence markers.
        lm_normalize : Optional[bool], default=True
            Indicates whether to normalize the text before scoring with the
            language model.
        """
        super().__init__(
            beam_size,
            tokenizer,
            inference,
            patience,
            None,
            lm_alpha,
            lm_beta,
            lm_eos,
            lm_normalize,
        )

        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the models, only once
        self.lm_model = (
            kenlm.Model(lm_path) if lm_path is not None else None
        )  # pylint: disable=c-extension-no-member
        if llm_path:
            self.llm_model = LLMSingleton.get_model(llm_path).to(self.device)
            self.llm_tokenizer = LLMSingleton.get_tokenizer(llm_path)
        else:
            self.llm_model = self.llm_tokenizer = None

    def lm_score_and_word_count(self, sequence) -> Tuple[float, int]:
        """Get n-gram and large language model scores.

        Parameters
        ----------
        sequence : tuple of int
            A sequence of token IDs.

        Returns
        -------
        float
            The n-gram language model score for the decoded text of the sequence.
        float
            The large language model score for the decoded text of the sequence.
        """
        # Convert sequence of tokens to text
        sequence = tuple(t for t in sequence if t not in self.special_tokens)
        if len(sequence) < LMOptions().lm_token_threshold:
            return None, 0.0
        text = self.tokenizer.decode(sequence)

        # Early return for empty text
        if not text:
            return None, 0.0
        logging.debug('LM&LLM text: "%s"', text)

        # Normalize the text
        if self.lm_normalize:
            normalized_text = self.lm_normalizer(text)
        else:
            normalized_text = text
        logging.debug('LM&LLM text normalized: "%s"', normalized_text)

        # Check for end of sentence and end of word:
        eos = text[-1] in self.lm_eos

        # word_count = len(normalized_text.split())
        # logging.debug("Word count: %d", word_count)

        # In KenLM, the most probable sequences have a higher score:
        score_lm = self.lm_model.score(normalized_text, bos=True, eos=eos)
        logging.debug("LM score: %f", score_lm)

        # Tokenize the input
        tokens = self.llm_tokenizer(normalized_text, return_tensors="pt").to(
            self.device
        )

        # Get input IDs and attention mask
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Calculate output from the model
        outputs = self.llm_model(
            input_ids, attention_mask=attention_mask, labels=input_ids
        )

        # Get the log probabilities of the last token
        log_probs = outputs.logits[:, -1, :].softmax(dim=-1)
        # Use the highest log probability as the score
        max_log_prob = log_probs.max().item()
        # Convert from natural log to log10 (like KenLM)
        score_llm = max_log_prob  # / math.log(10) * -100

        logging.debug("LLM score: %f", score_llm)

        return score_lm, score_llm


# Extending the DecodingTask class to support an BeamSearchWithLM
# ===============================================================


# Store a reference to the original __init__
original_decoding_task_init = DecodingTask.__init__


def new_decoding_task_init(self, model: Whisper, options: DecodingOptions):
    """Create the the DecodingTask class instance.

    This will replace the original constructor.

    Example
    -------
    >>> DecodingTask.__init__ = new_decoding_task_init
    """
    # Call the original constructor using the stored reference:
    original_decoding_task_init(self, model, options)

    # New logic:
    lm_options = LMOptions()
    if options.beam_size is not None:
        if lm_options.llm_path is not None and lm_options.lm_path is not None:
            logging.debug("Decoder: BeamSearchDecoderWithLMAndLLM")
            self.decoder = BeamSearchDecoderWithLMAndLLM(
                options.beam_size,
                self.tokenizer,
                self.inference,
                options.patience,
                lm_options.lm_path,
                lm_options.llm_path,
                lm_options.lm_alpha,
                lm_options.lm_beta,
                lm_options.lm_eos,
                lm_options.lm_normalize,
            )
        elif lm_options.llm_path is not None:
            logging.debug("Decoder: BeamSearchDecoderWithLLM")
            self.decoder = BeamSearchDecoderWithLLM(
                options.beam_size,
                self.tokenizer,
                self.inference,
                options.patience,
                lm_options.llm_path,
                lm_options.lm_alpha,
                lm_options.lm_beta,
                lm_options.lm_eos,
                lm_options.lm_normalize,
            )
        else:
            logging.debug("Decoder: BeamSearchDecoderWithLM")
            self.decoder = BeamSearchDecoderWithLM(
                options.beam_size,
                self.tokenizer,
                self.inference,
                options.patience,
                lm_options.lm_path,
                lm_options.lm_alpha,
                lm_options.lm_beta,
                lm_options.lm_eos,
                lm_options.lm_normalize,
            )


# Monkey patching the DecodingTask constructor:
DecodingTask.__init__ = new_decoding_task_init
