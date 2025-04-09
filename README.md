# Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages

[![arXiv - Paper](https://img.shields.io/badge/cs.CL-2503.23542-b31b1b?&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2503.23542)
[![Hugging Face Implementation](https://img.shields.io/badge/Code-🤗HuggingFace-yellow.svg)](https://github.com/hitz-zentroa/whisper-lm-transformers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Welcome to the repository for "Whisper-LM," an extension to OpenAI's Whisper
models that integrates n-gram and large language models (LM) to enhance
automatic speech recognition (ASR) performance, particularly for low-resource
languages. This repository contains scripts and tools used in our research and
can also be adapted for other languages or models.

For those looking to fine-tune Whisper models specifically, we recommend
starting with
[the Whisper Fine-Tuning Event scripts provided by Hugging Face](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event).
However, feel free to use your own fine-tuned models with this code.

## Using LMs with Transformers Library

For users interested in a transformers library-compatible implementation, visit
[whisper-lm-transformers](https://github.com/hitz-zentroa/whisper-lm-transformers).
That repository reimplements the functionality using the transformers Whisper
model, and is more user friendly. Note that results may vary slightly due to
minor differences in internal workings.

## Install Requirements

For result reproduction, make sure the correct version of Whisper is installed,
and use the requirements file used with Python 3.8:

```shell
pip install -U openai_whisper==20230918
pip install -r requirements.txt
```

If your need a more customized installation, these are the required packages:

* `datasets`
* `jiwer`
* `kenlm==0.2.0`
* `librosa`
* `numpy`
* `openai_whisper==20231117`
* `optuna`: used by the `lm_optimize.py` script.
* `tabulate`
* `torch`
* `tqdm`
* `transformers`

## Quick Start

Here, we are going to perform a simple transcription using Whisper with an LM.

### Language Model Download

Start by downloading the desired LM:

```shell
wget -O 5gram-eu.bin https://aholab.ehu.eus/~xzuazo/models/Basque%20LMs/5gram.bin
```

### Using Fine-Tuned Models

This is step is only needed if you want to use a model in Hugging Face format.
It needs to be converted back to the OpenAI format before using it here.

For example, to use the fine-tuned Tiny size Whisper model from
[zuazo/whisper-tiny-eu](https://huggingface.co/zuazo/whisper-tiny-eu), we
convert it to Open AI format:

```shell
./convert_hf_to_openai.py \
    --checkpoint zuazo/whisper-tiny-eu \
    --whisper_dump_path zuazo-whisper-tiny-eu.pt
```

### Transcription Example

Finally, to perform a simple transcription using the converted model and an LM:

```python
>>> import whisper

>>> # Hack Whisper to support LM and load the options interface to set it up:
>>> from whisper_decoder_with_lm import LMOptions

>>> # Select an audio file:
>>> audio_path = "tests/fixtures/common_voice_eu_18591439.mp3"

>>> # Set original Whisper transcription options (this is important):
>>> decode_options = {
...     "language": "eu",
...     "without_timestamps": True,
...     "temperature": 0.0,
...     "beam_size": 5,
... }
>>> transcribe_options = {"task": "transcribe", **decode_options}

>>> # Set LM-specific options:
>>> LMOptions().lm_path = "5gram-eu.bin"
>>> LMOptions().lm_alpha = 0.33582369
>>> LMOptions().lm_beta = 0.68825565

>>> # Load the model and transcribe the audio:
>>> model = whisper.load_model("zuazo-whisper-tiny-eu.pt")
>>> result = model.transcribe(audio_path, **transcribe_options)

>>> result["text"]
'Non-demontre dago langarizoka eta non bolikosta?'

```

To use a large language model (LLM) we have the `llm_path` argument, with
exactly the same syntax, together with the same `lm_alpha` and `lm_beta`
parameters. This parameter supports Hugging Face model names:

```python
>>> # Set LLM-specific options:
>>> LMOptions().llm_path = "HiTZ/latxa-7b-v1.2"
>>> LMOptions().lm_alpha = 2.73329396
>>> LMOptions().lm_beta = 0.00178595

```

To see a more complete example of how to use an LM with Whisper, check the
[`whisper_evaluate.py`](https://github.com/hitz-zentroa/whisper-lm/blob/main/whisper_evaluate.py)
script that is used to generate the evaluations in Common Voice, or other
datasets hosted in Hugging Face. There is also the
[`whisper_evaluate_external.py`](https://github.com/hitz-zentroa/whisper-lm/blob/main/whisper_evaluate_external.py)
script that is used to evaluate the models in datasets outside the Hugging Face
Hub.

These are the n-gram language models used in the paper:

* Basque: [5gram-eu.bin](https://aholab.ehu.eus/~xzuazo/models/Basque%20LMs/5gram.bin)
* Galician: [5gram-gl-27M.bin](https://aholab.ehu.eus/~xzuazo/models/Other%20LMs/5gram-gl-27M.bin)
* Catalan: [5gram-ca-27M.bin](https://aholab.ehu.eus/~xzuazo/models/Other%20LMs/5gram-ca-27M.bin)
* Spanish: [5gram-es-27M.bin](https://aholab.ehu.eus/~xzuazo/models/Other%20LMs/5gram-es-27M.bin)

And these are the large language models:

* Basque: [HiTZ/latxa-7b-v1.2](https://huggingface.co/HiTZ/latxa-7b-v1.2)
* Galician: [proxectonos/Carballo-cerebras-1.3B](https://huggingface.co/proxectonos/Carballo-cerebras-1.3B)
* Catalan: [projecte-aina/FLOR-6.3B](https://huggingface.co/projecte-aina/FLOR-6.3B)
* Spanish: [projecte-aina/FLOR-6.3B](https://huggingface.co/projecte-aina/FLOR-6.3B)

## Using This Repository for Your Research

This codebase is structured to facilitate reproduction of our results and to
aid others in extending Whisper models with LMs for additional languages.
Here's how you can use this repository:

### Fine-Tuning Whisper

Instructions and scripts for fine-tuning are available
[here](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event).
This process should be done prior to LM integration if using non-English or underrepresented languages.

### Creating an N-Gram Language Model

Feel free to utilize our scripts to generate text corpora:

```shell
./lm_corpora_create.sh --lang eu --opusall corpora-eu.txt
```

and then build language models using KenLM:

```shell
make LLANG=eu lm
```

or [create your own KenLM model](https://kheafield.com/code/kenlm/).

Keep in mind that the quality of the texts used to create the language-model
considerably affect its effectiviness.

### Optimizing Language Model Parameters

Optimize the alpha and beta parameters for the LM:

```shell
./lm_optimizer.py "zuazo-whisper-tiny-eu.pt" \
    --dataset_split "train+validation" \
    --dataset_name "eu" \
    --language "eu" \
    --beam_size 5 \
    --lm_path "5gram-eu.bin" \
    --n_trials 100
    --journal_storage \
    --n_jobs 32
```

We can also optimize for a large language models using the `--llm_path`
argument:

```shell
./lm_optimizer.py "zuazo-whisper-tiny-eu.pt" \
    --dataset_split "train+validation" \
    --dataset_name "eu" \
    --dataset_shuffle 'True' \
    --dataset_n 4000 \
    --language "eu" \
    --beam_size 5 \
    --batch_size 16 \
    --llm_path "HiTZ/latxa-7b-v1.2" \
    --lm_alpha_min 0 --lm_beta_min 0 \
    --lm_alpha_max 3 --lm_beta_max 3 \
    --n_trials 100 \
    --journal_storage \
    --n_jobs 1
```

In this case, we will limit the jobs to 1 per GPU, because we are loading both
the Whisper and the LLM model in the GPU memory. This was run in 7 NVIDIA
A100-SXM4-80GB GPU.

### Evaluating Models

Evaluate the performance on standard datasets or your own data:

```shell
./whisper_evaluate.py "zuazo-whisper-tiny-eu.pt" \
    --dataset "mozilla-foundation/common_voice_13_0" \
    --dataset_name "eu" \
    --dataset_split "test" \
    --language "eu" \
    --beam_size 5 \
    --lm_path "5gram-eu.bin" \
    --lm_alpha 0.33582369 --lm_beta 0.68825565
```

If the dataset is not in Hugging Face, we can use the
`whisper_evaluate_external.py` script:

```shell
./whisper_evaluate_external.py "zuazo-whisper-tiny-eu.pt" \
    ~/ahomytts \
    --language "eu" \
    --lm_path "5gram-eu.bin" \
    --lm_alpha 0.33582369 --lm_beta 0.68825565 \
    --beam_size 5
```

The dataset is expected to have the transcriptions in `*.txt` files with the
same name as the audio files.

### Notebooks

In the `notebooks/` directory is the code used to generate the tables and
plots of the article.

## Contributing

Contributions are welcome! Please refer to
[CONTRIBUTING.md](https://github.com/hitz-zentroa/whisper-lm/blob/master/CONTRIBUTING.md)
for guidelines on how to propose improvements, report issues, or submit pull requests.

## Citation

If you find this helpful in your research, please cite:

```bibtex
@misc{dezuazo2025whisperlmimprovingasrmodels,
      title={Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages},
      author={Xabier de Zuazo and Eva Navas and Ibon Saratxaga and Inma Hernáez Rioja},
      year={2025},
      eprint={2503.23542},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.23542},
}
```

Please, check the related paper preprint in
[arXiv:2503.23542](https://arxiv.org/abs/2503.23542)
for more details.
