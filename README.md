# Flowspeech
Project to migrate Flowchase speech tech from octave to python

First, clone this repo

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

This is needed for pyworld library (f0 extraction):
```
sudo apt install g++
```

Install python packages:
```
pip install -r requirements.txt
```

Check it works with `python` and then `from speech_tech import *`

To be able to use mp3 files with librosa library:
```
conda install -c conda-forge ffmpeg
```

## Make dirs for input and result files
```
mkdir inputs
mkdir results
```

## Test modules

```
pytest
```

## Data
```
cd ..
git clone https://github.com/flowchase/audio-with-analysis-ids
```

You can use ```get_data()``` function.

## Server

A Flask API is provided to access the modules.
Run ```sh run_server.sh``` to launch it.
The process is in two steps (post requests)

- Upload an audio file (see index.html for an example of post request)
- call a module with the filename that will return a result (see dummy_client.py for an example of post request)

Create a folder set to receive the uploaded files:
```
mkdir upload_files
```

## Docker application
You can also build the Dockerfile that will install everything and serve the application with Flask with nginx backend.
I used this info to do that: 
https://github.com/srcecde/flask-docker-ec2
<!-- https://github.com/ram-ch/Building-microservices-with-docker-on-AWS -->

```
sudo docker-compose up -d
```

On AWS, I chose an Amazon Linux 2 with Docker installed. 
Check the command to ssh to it on AWS.
```
ssh -i "flowspeech.pem" ec2-user@ec2-52-47-122-20.eu-west-3.compute.amazonaws.com
```

But I had to install docker-compose like this:
```
pip install docker-compose
```

change the line of nginx/web.conf
"	proxy_pass  http://aws.server.ip.here:5000/;"


If you just want to use it locally, without nginx server, you can build only flowspeech image:
```
docker build -t flowspeech .
docker run -d -p 8000:8000 flowspeech
```