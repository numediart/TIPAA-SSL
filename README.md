# Flowspeech
Project to migrate Flowchase speech tech from octave to python

## Web service

A Flask API is provided to access the modules.

Examples of client requests are available in ```dummy_client.py``` and ```index.html```

The process is in two post requests:

### Upload an audio file with arguments:
Its filename should be unique not to overwrite a previous one. Maybe It would be better to merge both steps in one post request to avoid this problem ?

```
url= "/"
files = {'file': file}
```
Maybe the example in ```index.html``` with a form is better for you:
```
<form method="POST" action="" enctype="multipart/form-data">
    <p><input type="file" name="file"></p>
    <p><input type="submit" value="Submit"></p>
</form>
```


### Call a module with the audio filename and sentenceID with arguments:
```
module="sentenceStress" # or "wordStress", "iContrast"
url= "/flowspeech/"+module
data={"sentenceID":str(sentenceID), "filename":filename}
```
An example of feedback for each module:

- sentenceStress:  the feedback is a list of 0/1 for each word, the 1 being the sentence stress
For the sentence "I would love to go to ireland", the correct answer would be:
```
b'{"status": "success", "result": [0, 0, 1, 0, 0, 0, 0]}'
```

- wordStress: the feedback is a list of 0/1 for each syllable, the 1 being the word stress
```
b'{"status": "success", "result": [1, 0]}'
```

- iContrast: the feedback is 0 if the phoneme is "IH" (short i) and 1 if the pohneme is "IY" (long i).
For the word "slip", 
```
b'{"status": "success", "result": 0}'
```

- example if error:
```
b'{"status": "error: ./upload_files/iC_111_sl.wav could not be loaded", "result": []}'
```



### Lower level functions
Besides existing module, I am working on two lower level functions. 

The logic behind them is to use text and audio as input. The text is automatically phonetized and "grammarized", then htk model is used and:
- vowelStresses gives stress scores for each syllable of each word between 0 and 1
```
url='/vowel_stresses'
data={"text":text, 'filename':filename}
```
Example of output:
```
 b'{"status": 0, "result": [[0.8357734306528976], [0.49237615361307363], [0.8516958573738025], [0.2795877688695186], [0.5738685725256959], [0.27777777777777773], [0.7328439332799466, 0.4460136293979836, 0.6261916617539647]]}'
 ```

 if you pass only one word with only one syllable, the result will be a `[[nan]]`

- phonemeContrast gives you a detected transcription based on a target phoneme and a set of alternatives (in CMU phonemes)
```
url='/phonemeContrast'
data={"text":text, 'filename':filename, 'word_id':word_id, 'alternatives':alternatives, 'target':target}
```
Example of output:
```
b'{"status": "success", "result": ["T", "ER1", "N", "D AH0"], "ground_truth": ["T", "ER1", "N", "D"]}'
```

## Docker application
You can also build the Dockerfile that will install everything and serve the application with Flask with nginx backend.
I used this info to do that: 
https://github.com/srcecde/flask-docker-ec2
<!-- https://github.com/ram-ch/Building-microservices-with-docker-on-AWS -->

First, clone this repo, then in it:

```
sudo docker-compose up -d
```

On AWS, I chose an Amazon Linux 2 with Docker installed. 
Check the command to ssh to it on AWS.
```
ssh -i "~/flowspeech.pem" ec2-user@ec2-52-47-122-20.eu-west-3.compute.amazonaws.com
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

To git pull inside a container:
```
docker exec flaskapp git pull
```
Connect to a bash terminal without affecting the running state. 
```
docker exec -it flaskapp bash
```
## Manual Installation
### Install HTK

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

## Test modules

```
pytest
```

## Data
Data of actor recordings with sentenceID etc.
```
cd ..
git clone https://github.com/flowchase/audio-with-analysis-ids
```

You can use ```get_data()``` function.
