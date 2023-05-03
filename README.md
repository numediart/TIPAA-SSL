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
- content_tools contains simple web apps related exploring phonetic dictionaries, using linguistic data extraction (phonetization + syllabification in text and phonetics) and other related content processing tools



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

### Install miniconda and python dependencies
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Then follow instructions and open a new terminal, conda will be activated

```
conda create -n flowspeech python=3
conda activate flowspeech
```

Then basically follow the same installation commands described in the Dockerfile, but directly on your linux machine


## Data

### Data of actor recordings
Go in this [drive folder](https://drive.google.com/drive/folders/1-c34uCaNL8PvokYFPaqWYin6FGPXq-3t?usp=share_link)

- Download the folder "audio-with-analysis-ids" 
Paste it inside "data/" folder of this repo.
You can now use ```get_data()``` function in "src/label_data_processing.py" now.
- Create a folder called "synth_audio" in "data/" and drop the folder "cmu_words" in it after extracting the zip
- Drop "flwc-phrase-audios" after extracting the zip


### Librispeech data
https://www.openslr.org/12/

I use dev-clean and test-clean sets.
```
cd data
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar xvfz dev-clean.tar.gz
rm dev-clean.tar.gz 
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar xvfz test-clean.tar.gz
rm test-clean.tar.gz 
cd ..
```

Get phonetic alignments data:
```
sudo apt install unzip
cd data
mkdir librispeech_alignments
cd librispeech_alignments
curl https://zenodo.org/record/2619474/files/librispeech_alignments.zip?download=1 --output librispeech_alignments.zip
unzip librispeech_alignments.zip
rm librispeech_alignments.zip
cd ..
cd ..
```

### MAILABS dataset

Download data archives from [MAILABS website](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/#:~:text=Statistics%20%26%20Download%20Links)
Download the ones with the tags "en_UK", "en_US" in priority. (You can also download "es_ES" and "fr_FR" for experimenting later, but not necessary right now)

Commands for doing it for "en_UK" if you are at the root of the repo:
```
cd data
mkdir MAILABS
cd MAILABS
wget https://data.solak.de/data/Training/stt_tts/en_UK.tgz
tar xvfz en_UK.tgz
rm en_UK.tgz
cd ../..
```

### Other sets of data

There are other useful sets of data, we don't have a place for them online yet. Ask Noé

## Test modules

```
pytest test_DL_modules.py
pytest test_DL_api.py
```
