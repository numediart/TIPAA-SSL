# Flowspeech

Flowchase's speech tech

## Structure of the repository

- DL_speech_tech is the "center" of the repo and contains the pronunciation aspect function with intermediary steps that are common between different pronunciation aspects
  - It depends on "src/" folder which contains the part of the code that is used "online" (used in production behind the API). It contains:
    - text and phonetics processing, pronunciation dictionaries, audio processing, ...
    - wav2vec2_frame_prediction that contins a class that wraps the model pipeline of wav2vec2 finetuned model, dimension reduction, frame classification, and dtw forced alignment (itself defined in another class)
    - some data processing and loading functions and other utilities
- server folder contains the definition of the API endpoints that call the DL_speech_tech functions
- nginx is a docker image of a server making the link between flask and the internet (just for deployment)
- data folder containing different sets of data tha has to be populated (see instructions below)
- scripts folder contains scripts that are used "offline", i.e. not used behind the API endpoints. E.g. script for
  - download pretrained models
  - extract forced alignments, i.e. time-aligned phonetic transcriptions, thanks to "Montreal Forced Aligner" (MFA)
  - experiment different frame reduction and frame classifiers (where a frame is an output vector of the wav2vec2 model)
  - ...
- content_tools contains web apps related to exploring phonetic dictionaries, using linguistic data extraction (phonetization + syllabification in text and phonetics) and other related content processing tools

## Web service

A Flask API is provided to access the speech analysis features by pronunciation aspect.
**To learn more about making requests to the Speech Processing API**, please see [this documentation file](/API_v2.md).

## Docker application on server: deployment and update instructions

Please see [this documentation file](/DEPLOYMENT.md).

## Local Installation

### Download codes and models

```
git clone https://github.com/flowchase/flowspeech
cd flowspeech
git clone https://github.com/noetits/charsiu
sudo apt-get install git-lfs
git lfs install
python scripts/download_models.py
```

These will clone necessary repositories and install git-lfs (lfs stand for large file storage) in order to download some open source deep learning models.

At the root of the "flowspeech" repository, create a folder named "models", and drop in it the folder "model_mailabs_pca_0.95_knn_10_w" that is in [Flowchase's drive storage](https://drive.google.com/drive/folders/1drAmLjPOsl1QrfEuOybmiv-R-Sxd-yFC?usp=share_link)

After executing the `scripts/download_models.py` script, you should have a few pretrained models inside "hf_models" folder.
The Flowchase drive folder above contains a file named `last_hidden_state.quant.onnx`. This is a compressed version of a pretrained huggingface model, and should go into the corresponding folder which is "hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft"

### Install micromamba and python dependencies

<!-- ```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Then follow instructions and open a new terminal, conda will be activated

```
conda create -n flowspeech python=3
conda activate flowspeech
``` -->

<!-- Basically follow the same installation commands described in the Dockerfile, but directly on your linux machine -->

First install all necessary apt packages listed in [./Dockerfile](/Dockerfile#L6)

(Last time I tried, on a WSL Ubuntu, I just needed to apt install g++ festival espeak-ng)

To setup the micromamba environment, you first have to install it.
Install micromamba:
https://mamba.readthedocs.io/en/latest/micromamba-installation.html

```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

```
micromamba create -n flowspeech_mm
micromamba activate flowspeech_mm
```

Then you can proceed to install dependencies. It is easier to install the conda package of montreal-forced-aligner rather than pursuing the same commands as in the Dockerfile. But it takes more space on your disk (which is not a big deal for a local installation but it is of the docker image).

```
micromamba install kaldi=*=*cpu* montreal-forced-aligner
```

Once MFA is installed you can install the rest of this repository dependencies.

```
micromamba install -y -n flowspeech_mm -f env.yml
```

## Test modules

The unit tests should work after these step

```
pytest test_DL_modules.py
pytest test_DL_api.py
pytest test_models.py
```
