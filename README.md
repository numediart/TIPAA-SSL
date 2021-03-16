# Flowspeech
Project to migrate Flowchase speech tech from octave to python

## Install HTK

Either use the Dockerfile from https://github.com/loretoparisi/htk 

or to install directly on e.g. Ubuntu 20.04:
```
sudo apt-get update && sudo apt-get install -y      libc6-dev-i386      libx11-dev      gawk      curl      git
sudo apt install make
git clone https://github.com/loretoparisi/htk
cd htk
./configure --disable-hslab &&      sudo make all &&      sudo make install
```

Check it works calling `HVite` command

## Install miniconda and python dependencies
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Then follow instructions and open a new terminal, conda will be activated

```
conda create -n flowspeech python=3
conda activate flowspeech
```

This is needed for pyworld library:
```
sudo apt install g++
```

Install python packages:
```
pip install -r requirements.txt
```

Check it works with `python` and then `from speech_tech import *`


## Make dirs for input and result files
```
mkdir inputs
mkdir results
```

## Test modules

```
pytest
```
