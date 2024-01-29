# Flowspeech

Flowchase's speech tech

## Structure of the repository

- The `flowspeech` package contains the actual implementation of the tech:
  - `DL_speech_tech` contains the pronunciation aspect function with intermediary steps that are common between different pronunciation aspects
  - text and phonetics processing, pronunciation dictionaries, audio processing, ...
  - `wav2vec2_frame_prediction` that contains a class that wraps the model pipeline of wav2vec2 finetuned model, dimension reduction, frame classification, and dtw forced alignment (itself defined in another class)
  - some data processing and loading functions and other utilities
- `app` contains the definition of the API endpoints that call the `DL_speech_tech` functions
- `nginx` contains the configuration and Docker files for the deployment server
- `data` contains different datasets, and also has to be populated after cloning the repository (see [here](https://www.notion.so/flowchase/Speech-datasets-818af2e4ea1749469194642f7226c0ff))
- `models` contains trained model checkpoints for the part of the processing pipeline which we train ourselves (dimensionality reduction + frame classifier)
- `syllabipy` contains a tweaked version of the syllabipy package, used to syllabify sentences
- `scripts` folder contains scripts that are used "offline", i.e. not used behind the API endpoints. E.g. script for
  - download pretrained models
  - extract forced alignments, i.e. time-aligned phonetic transcriptions, thanks to "Montreal Forced Aligner" (MFA)
  - experiment different frame reduction and frame classifiers (where a frame is an output vector of the wav2vec2 model)
  - ...
- `content_tools` contains a simple streamlit-based web interface related to exploring phonetic dictionaries, using linguistic data extraction (phonetization + syllabification in text and phonetics) and other related content processing tools

## Web service

A Flask API is provided to access the speech analysis features by pronunciation aspect.
**To learn more about making requests to the Speech Processing API**, please see [this documentation file](/API_v2.md).

## Docker application on server: deployment and update instructions

Please see [this documentation file](/DEPLOYMENT.md).

## Local Installation

### Download codes and models

```
git clone https://github.com/flowchase/flowspeech
```

The [Flowchase drive folder](https://drive.google.com/drive/folders/1drAmLjPOsl1QrfEuOybmiv-R-Sxd-yFC?usp=share_link) contains a file named `last_hidden_state.quant.onnx`.
This is a compressed version of a pretrained wav2vec2 model from hugginface.
Create a folder `hf_models` and download that file into it.
Note: the folder also contains a backup of the dimensionality reduction and frame classifiers that run on top of the base wav2vec2 model, but they are now commited in this repository in [models](./models).

### Install micromamba and python dependencies

First install all necessary packages listed in [./Dockerfile](/Dockerfile#L12):

```bash
sudo apt-get install git-lfs espeak-ng festival
git lfs install
```

To setup the micromamba environment, you first have to install micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

```
micromamba create -n flowspeech_mm -f env.yml
micromamba activate flowspeech_mm
pip install -e .
```

### Notes for MacOS and Apple M1/... chips

A few adjustments are needed to be able to run on Apple Silicon, as some of the dependencies
on conda are not available for ARM architectures.

Install basic dependencies:

```bash
brew install git-lfs micromamba
```

Install homebrew for x86 and install dependencies:

```bash
arch -x86_64 zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install espeak
exit
```

Setup the conda environment for x86:

```bash
CONDA_SUBDIR=osx-64 micromamba create -n flowspeech_mm -f env.yml
micromamba activate flowspeech_mm
conda env config vars set PHONEMIZER_ESPEAK_PATH=/usr/local/bin/espeak
conda env config vars set PHONEMIZER_ESPEAK_LIBRARY=/usr/local/lib/libespeak.dylib
```

## Test modules

The unit tests should work after these step

```

pytest test_DL_modules.py
pytest test_DL_api.py
pytest test_models.py

```

## Download datasets

To test the tech's performance, develop new features, ..., you will need speech datasets.
Have a look at this page for instructions to obtain them: https://www.notion.so/flowchase/Speech-datasets-818af2e4ea1749469194642f7226c0ff

```

```
