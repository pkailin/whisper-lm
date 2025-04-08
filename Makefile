.PHONY: all test test_req black nb_black isort nb_isort pylint nb_pylint flake8 nb_flake8 autopep8 pydocstyle nb_pydocstyle ruff bandit autoflake pydocstringformatter style nb_style doctest kenlm kenlm_reduced lm lm_reduced install install-dev

# Targets to create help running the tests and create the language model.
#
# Example
# -------
# To create the language model limited to 27M sentences:
# ```
# $ make LLANG=eu lm_reduced
# ```
#
# This will create the language model for the Basque language.
#
# To run the Python style and unit tests:
# ```
# $ make test
# ```
#
# To run only the style tests:
# ```
# $ make style
# ```
#
# The notebook style tests are in `nb_tests`.
#
# To run only the unit tests:
# ```
# $ make unit
# ```

# Language used for the corpora:
LLANG := eu
# Include: Wikipedia, Opus, Extra corpus.
CORPORA := WOE
# Disable this for languages with big text resources:
OPUSALL := --opusall
# Corpus limited to this million of sentences (used by lm_reduced):
REDUCED := 27

# File to be tested in style tests:
FILES = *.py
NOTEBOOKS = notebooks/*.ipynb

# Default target that builds the language model, runs style checks, and tests:
all: style test

# Download Whisper Tiny and Medium models used in the tests:
zuazo-whisper-tiny-eu.pt:
	@echo "Downloading Whisper Tiny Model..."
	./convert_hf_to_openai.py \
	    --checkpoint zuazo/whisper-tiny-eu \
	    --whisper_dump_path zuazo-whisper-tiny-eu.pt

zuazo-whisper-medium-eu.pt:
	@echo "Downloading Whisper Medium Model..."
	./convert_hf_to_openai.py \
	    --checkpoint zuazo/whisper-medium-eu \
	    --whisper_dump_path zuazo-whisper-medium-eu.pt

# Prepare environment for testing by downloading necessary models and LMs:
test_req: zuazo-whisper-tiny-eu.pt zuazo-whisper-medium-eu.pt
	# This replaces the 5gram-$(LLANG).bin target during tests
	@echo "Downloading LM..."
	if [ ! -e 5gram-eu.bin ]; then \
	    wget -O 5gram-eu.bin https://aholab.ehu.eus/~xzuazo/models/Basque%20LMs/5gram.bin ; \
	fi

# Run all style checks and tests:
test: style nb_style unit doctest

# Run style checks:
style: black isort flake8 autopep8 pydocstyle ruff bandit autoflake pydocstringformatter pylint

# Run style checks for notebooks:
nb_style: nb_black nb_isort nb_flake8 nb_pydocstyle  # nb_pylint

# Check Python files using black:
black:
	@echo "Checking black..."
	black --check $(FILES)
	@echo

# Check Jupyter notebooks using black:
nb_black:
	@echo "Checking notebooks with black..."
	nbqa black --check $(NOTEBOOKS)
	@echo

# Sort imports in Python files using isort:
isort:
	@echo "Checking isort..."
	isort --profile=black --check $(FILES)
	@echo

# Sort imports in Jupyter notebooks using isort:
nb_isort:
	@echo "Checking notebooks with isort..."
	nbqa isort --profile=black --check $(NOTEBOOKS)
	@echo

# Check Python files using pylint:
pylint:
	@echo "Checking pylint..."
	pylint $(FILES)
	@echo

# Check Jupyter notebooks using pylint:
nb_pylint:
	@echo "Checking notebooks with pylint..."
	nbqa pylint $(NOTEBOOKS)
	@echo

# Check Python files using flake8:
flake8:
	@echo "Checking flake8..."
	flake8 $(FILES)
	@echo

# Check Jupyter notebooks using flake8:
nb_flake8:
	@echo "Checking notebooks with flake8..."
	nbqa flake8 $(NOTEBOOKS)
	@echo

# Auto-format Python files using autopep8:
autopep8:
	@echo "Checking autopep8 for changes..."
	@diffout=$$(autopep8 --diff $(FILES)); \
	if [ -z "$$diffout" ]; then \
	    echo "No changes needed!"; \
	else \
	    echo "Code formatting needed. Here is the diff:"; \
	    echo "$$diffout"; \
	    exit 1; \
	fi
	@echo

# Check Python files for docstring style using pydocstyle:
pydocstyle:
	@echo "Checking pydocstyle..."
	pydocstyle --convention=numpy __init__.py lm_optimizer.py whisper_decoder_with_lm.py whisper_evaluate_external.py whisper_evaluate.py
	pydocstyle --convention=google convert_hf_to_openai.py
	@echo

# Check Jupyter notebooks for docstring style using pydocstyle:
nb_pydocstyle:
	@echo "Checking notebooks with pydocstyle..."
	nbqa pydocstyle --convention=numpy $(NOTEBOOKS)
	@echo

# Check Python files for basic syntax errors using ruff:
ruff:
	@echo "Checking ruff..."
	ruff check $(FILES)
	@echo

# Perform security checks on Python files using bandit:
bandit:
	@echo "Checking bandit..."
	bandit src/
	@echo

# Remove unused imports from Python files using autoflake:
autoflake:
	@echo "Checking autoflake..."
	autoflake $(FILES)
	@echo

# Format Python docstrings using pydocstringformatter:
pydocstringformatter:
	@echo "Checking pydocstringformatter..."
	pydocstringformatter $(FILES)
	@echo

# Run unit tests:
unit: test_req
	@echo "Running pytest..."
	python -m pytest
	@echo

# Run doctests:
doctest: test_req
	@echo "Running doctest with pytest..."
	python -m doctest -v *.md
	@echo

# Install package for production use:
install:
	@echo "Installing required packages..."
	python -m pip install --upgrade pip
	python -m pip install -U openai_whisper==20230918
	python -m pip install -r requirements.txt
	@echo

# Install package for development use
install-dev: install
	@echo "Installing required dev package..."
	python -m pip install -r requirements_dev.txt
	@echo

# Create corpus files:
corpora-$(LLANG).txt:
	./lm_corpora_create.sh --lang $(LLANG) --corpora $(CORPORA) $(OPUSALL) corpora-$(LLANG).txt

# Create reduced corpus files (limited to 27M sentences):
corpora-$(LLANG)-$(REDUCED)M.txt: corpora-$(LLANG).txt
	head -n $(REDUCED)000000 corpora-$(LLANG).txt > corpora-$(LLANG)-$(REDUCED)M.txt
	@echo "Validating the number of lines in corpora-$(LLANG)-$(REDUCED)M.txt..."
	@if [ `wc -l < corpora-$(LLANG)-$(REDUCED)M.txt` -ne $(REDUCED)000000 ]; then \
	    echo "Error: The file corpora-$(LLANG)-$(REDUCED)M.txt does not have $(REDUCED)000000 lines."; \
	    rm -f corpora-$(LLANG)-$(REDUCED)M.txt ; \
	    exit 1; \
	else \
	    echo "Validation successful: The file has the correct number of lines."; \
	fi

# Build KenLM:
kenlm/build/bin/lmplz:
	wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
	mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j2

# Build ARPA file from full corpus:
5gram-$(LLANG).arpa: corpora-$(LLANG).txt
	kenlm/build/bin/lmplz -o 5 < corpora-$(LLANG).txt > 5gram-$(LLANG).arpa

# Build ARPA file from reduced corpus:
5gram-$(LLANG)-$(REDUCED)M.arpa: corpora-$(LLANG)-$(REDUCED)M.txt
	kenlm/build/bin/lmplz -o 5 < corpora-$(LLANG)-$(REDUCED)M.txt > 5gram-$(LLANG)-$(REDUCED)M.arpa

# Convert ARPA file to binary format:
5gram-$(LLANG).bin: 5gram-$(LLANG).arpa
	kenlm/build/bin/build_binary 5gram-$(LLANG).arpa 5gram-$(LLANG).bin

# Convert reduced ARPA file to binary format:
5gram-$(LLANG)-$(REDUCED)M.bin: 5gram-$(LLANG)-$(REDUCED)M.arpa
	kenlm/build/bin/build_binary 5gram-$(LLANG)-$(REDUCED)M.arpa 5gram-$(LLANG)-$(REDUCED)M.bin

# Create KenLM language model:
kenlm: kenlm/build/bin/lmplz 5gram-$(LLANG).bin

# Create KenLM language model from reduced corpus:
kenlm_reduced: kenlm/build/bin/lmplz 5gram-$(LLANG)-$(REDUCED)M.bin

# Build full language model:
lm: kenlm

# Build reduced language model:
lm_reduced: kenlm_reduced

# Clean up generated files during the LM generation:
clean:
	rm -f 5gram-$(LLANG).arpa 5gram-$(LLANG)_correct.arpa
	rm -f 5gram-$(LLANG)-$(REDUCED)M.arpa 5gram-$(LLANG)-$(REDUCED)M_correct.arpa
