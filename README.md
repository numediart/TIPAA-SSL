# Flowspeech

Flowchase's speech tech

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

Then basically follow the sme installation steps as in the Dockerfile

## Data

### Data of actor recordings with sentenceID etc.
```
cd data
git clone https://github.com/flowchase/audio-with-analysis-ids
cd ..
```

You can use ```get_data()``` function.

### Librispeech data:
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

### Other sets of data

There are other useful sets of data, we don't have a place for them online yet. Ask Noé

## Test modules

```
pytest test_DL_modules.py
pytest test_DL_api.py
```
