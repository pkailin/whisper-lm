#!/usr/bin/env bash
# Script to create text corpora for a language.
#
# The text will be segmented to a sentence per line and normalized using
# commonvoice-utils (covo). If you want to skip the text normalization, just
# remove the `covo norm` pipe commands. The same applies for segmentation with
# the `covo segment` pipes.
#
# Corpora included:
# - Wikipedia.
# - OPUS: Tatoeba, TED, and GlobalVoices or all (--opusall).
# - Basque: EusCrawl V1.
# - Galician: SLI GalWeb 1.0.
# - Spanish: Multilingual LibriSpeech (MLS) LM Resource.
# - Portuguese: CC100-Portuguese Dataset
#
# Keep in mind that Common Voice transcriptions are not included here.
#
# Usage example:
#  $ ./lm_corpora_create.sh --lang eu --opusall corpora-eu.txt
#
# Note: This script installs a specific version of commonvoice-utils in your
# Python environment.
#
# Based on: https://gitlab.com/xzuazo/commonvoice-docker/-/blob/main/lm.sh

# Install the following covo version (required for pre-processing):
# COVO_REPO='git+https://github.com/ftyers/commonvoice-utils@c25cc3b4688baa693d4c720c4fee66e30934b563'
COVO_REPO='git+https://github.com/zuazo-forks/commonvoice-utils@f51d68bdeb1c4bed5755ba0e1f20df409c2b9c95'  # pt

# Default configuration options (can be changed by command line arguments):
LLENGUA='eu'
CORPORA='WOE'  # Include: Wikipedia, Opus, Extra corpus.
OPUS_ALL='false'  # Whether to include all Opus or just the best.


syntax() {
  echo 'Syntax:' >&2
  echo "  $0 [--lang LANG] [--corpora CORPORA] [--opusall] OUTPUT.txt" >&2
}

usage() {
  syntax
  exit 255
}


DIR="$(dirname "$(readlink -f "${0}")")"
BASENAME="$(basename "${0}")"

# Parse command line arguments:
while [ "$#" -gt 0 ]
do
  case "${1}" in
    -l|--lang)
      LLENGUA="${2}"
      shift 2
      ;;
    -co|--corpora)
      CORPORA="${2}"
      shift 2
      ;;
    -a|--opusall)
      OPUS_ALL='true'
      shift 1
      ;;
    -h|--help)
      syntax
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

OF="${1}"
[ x"${OF}" = x ] && usage
echo "Output file: ${OF}"

TMP_DATA=".${BASENAME%.*}/${LLENGUA}"
echo "Creating temporary data directory: ${TMP_DATA}"
mkdir -p "${TMP_DATA}"

echo 'Installing requirements:'
pip uninstall -y commonvoice-utils
pip install -U "${COVO_REPO}" || exit 255

set -e  # exit on error
echo '========================================================================'

XLENGUA=$(echo ${LLENGUA} | cut -f1 -d'-')
OF_URLS="${TMP_DATA}/urls.txt"

: > "${OF}"  # empty the output file
: > "${OF_URLS}"

# Specific corpus for the language (if available):
if echo "${CORPORA}" | grep -qF 'E'
then
  # Multilingual LibriSpeech MLS (Spanish):
  echo 'Download an specific corpus (for some languages):'
  if [[ "${XLENGUA}" = 'es' ]]
  then
    wget -O - 'https://dl.fbaipublicfiles.com/mls/mls_lm_spanish.tar.gz' \
      | tar xOzf - 'mls_lm_spanish/data.txt' \
      | covo norm "${LLENGUA}" >> "${OF}" || exit 255
    echo 'LibriSpeech:'
    wc "${OF}"
  fi

  # EusCrawl (Basque):
  if [[ "${XLENGUA}" = 'eu' ]]
  then
    wget -O - 'http://ixa.ehu.eus/euscrawl/files/euscrawl-v1-free-txt.tar.bz2' \
      | tar xOjf - \
      | sed "s/$(echo -e "\xc2\xad")//g" \
      | sed "s/$(echo -e "\xef\xbb\xbf")//g" \
      | covo segment "${LLENGUA}" \
      | covo norm "${LLENGUA}" >> "${OF}" || exit 255
    echo 'EusCrawl:'
    wc "${OF}"
  fi

  # SLI GalWeb (Galician):
  if [[ "${XLENGUA}" = 'gl' ]]
  then
    echo 'Downloading SLI GalWeb Corpus...'
    if [ ! -f "${TMP_DATA}/SLI_GalWeb.1.0/SLI_GalWeb.1.0" ]
    then
      if [ ! -f "${TMP_DATA}/SLI_GalWeb.1.0.tar.gz" ]
      then
        for i in $(seq -w 0 12)
        do
          wget -O - "https://github.com/xavier-gz/SLI_Galician_Corpora/raw/main/SLI_GalWeb.1.0/SLI_GalWeb.1.0.tar.gz.part${i}" \
            || exit 255
        done > "${TMP_DATA}/SLI_GalWeb.1.0.tar.gz"
      fi
      tar xvzf "${TMP_DATA}/SLI_GalWeb.1.0.tar.gz" -C "${TMP_DATA}" || exit 255
    fi

    if [ -f "${TMP_DATA}/SLI_GalWeb.1.0/SLI_GalWeb.1.0" ]
    then
      cat "${TMP_DATA}/SLI_GalWeb.1.0/SLI_GalWeb.1.0"  \
        | covo segment "${LLENGUA}" \
        | covo norm "${LLENGUA}" >> "${OF}" || exit 255
      echo 'SLI_GalWeb:'
      wc "${OF}"
    fi
  fi

  # Catalan Textual Corpus (Catalan):
  if [[ "${XLENGUA}" = 'ca' ]]
  then
    echo 'Downloading Catalan Textual Corpus...'
    if [ ! -f "${TMP_DATA}/catalan_textual_corpus.zip" ]
    then
      wget -O "${TMP_DATA}/catalan_textual_corpus.zip" \
        'https://zenodo.org/record/4519349/files/catalan_textual_corpus.zip?download=1'
    fi
    unzip -p "${TMP_DATA}/catalan_textual_corpus.zip" \
      'corpus/catalan_textual_corpus.txt' \
      | covo norm "${LLENGUA}" >> "${OF}" || exit 255
    echo 'Catalan Textual Corpus:'
    wc "${OF}"
  fi

  if [[ "${XLENGUA}" = 'pt' ]]
  then
    wget -O - 'https://dl.fbaipublicfiles.com/mls/mls_lm_portuguese.tar.gz' \
      | tar xOzf - 'mls_lm_portuguese/data.txt' \
      | covo norm "${LLENGUA}" >> "${OF}" || exit 255
    echo 'LibriSpeech:'
    wc "${OF}"
  fi
  # if [[ x"${XLENGUA}" = x'pt' ]]
  # then
  #   wget -c -O "${TMP_DATA}/pt.txt.xz" 'http://data.statmt.org/cc-100/pt.txt.xz'
  #   xzcat "${TMP_DATA}/pt.txt.xz" \
  #     | covo norm "${LLENGUA}" >> "${OF}"
  #   echo 'CC100-Portuguese Dataset:'
  #   wc "${OF}"
  # fi
fi

# Wikipedia:
if echo "${CORPORA}" | grep -qF 'W'
then
  echo 'Download the latest Wikipedia dump:'
  wget \
    --directory-prefix="${TMP_DATA}" \
    --continue \
    -O "${TMP_DATA}/${XLENGUA}wiki-latest-pages-articles.xml.bz2" \
    "https://dumps.wikimedia.org/${XLENGUA}wiki/latest/${XLENGUA}wiki-latest-pages-articles.xml.bz2" \
    || exit 255
  covo dump "${TMP_DATA}/${XLENGUA}wiki-latest-pages-articles.xml.bz2" \
    | covo segment "${LLENGUA}" | covo norm "${LLENGUA}" >> "${OF}" || exit 255
  echo 'Wikipedia:'
  wc "${OF}"
fi

# OPUS Corpus:
if echo "${CORPORA}" | grep -qF 'O'
then
  echo 'Download data from OPUS corpus:'
  covo opus "${XLENGUA}" \
    | grep -e Tatoeba -e OpenSubtitles -e TED -e GlobalVoices \
    | cut -f2 > "${OF_URLS}"
  LINES="$(cat "${OF_URLS}" | wc -l)"
  if [[ "${LINES}" -gt 0 ]]
  then
    cat "${OF_URLS}" | xargs wget -O - | zcat \
      | covo norm "${LLENGUA}" >> "${OF}"
    echo 'OPUS:'
    wc "${OF}"
  fi
  if [[ "${OPUS_ALL}" = 'true' ]]
  then
    covo opus $XLENGUA | cut -f2 > "${OF_URLS}"
    LINES=$(cat "${OF_URLS}" | wc -l)
    if [[ ${LINES} -gt 0 ]]; then
      cat "${OF_URLS}" | xargs wget -O - | zcat \
        | covo norm "${LLENGUA}" >> "${OF}"
      echo 'OPUS2:'
      wc "${OF}"
    fi
  fi
fi

echo "Output file: ${OF}"
echo 'TOTAL:'
wc "${OF}"
