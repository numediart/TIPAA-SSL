# Flowspeech
Project to migrate Flowchase speech tech from octave to python

## Web service

A Flask API is provided to access the modules.

Examples of client requests are available in ```dummy_client.py```

The process is in two post requests: 
- Upload an audio file, 
- Call a module, or a lower level function

### Upload an audio file with arguments:

```
url= "/upload"
files = {'file': file}
```
The server returns a random ID "rID" to be used for processing afterwards.

### Call a module with arguments:
```
module="sentenceStress" # or "wordStress"
url= "/flowspeech/"+module
data={"phonetics":phonetics, "rID":rID}
```
The phonetics is consituted of CMU phonemes with seperators for phonemes ("_"), syllables ("|") and words (" ").

An example of feedback for each module:

- sentenceStress:  the feedback is 
    - status ("success" or "error: ...") this success means a technical success in the sense that there were no failure, but not that the recognition was successful
    - a list of stress intensities between 0 and 100 for each word
    - a list of 0/1 for each word, the 1 being the sentence stress

For the sentence "I would love to go to ireland", 
input:
```
data={"phonetics":"AY1 W_UH1_D L_AH1_V T_UW1 G_OW1 T_UW1 AY1|ER0|L_AH0_N_D", "rID":'487c3fe1-5f17-4010-a019-92b1c6ebfc5a'}
```


the correct answer would be:
```
b'{"status": "success", "stress_intensities": [83, 49, 85, 27, 57, 27, 73], "stress_binaries": [0, 0, 1, 0, 0, 0, 0]}'
```

- wordStress: the feedback is a 
    - status ("success" or "error: ...")
    - a list of stress intensities for each syllable of each word between 0 and 100 by word 
    - a list of 0/1 for each syllable of each word, the 1 being the word stress

For the word "toothpaste", input:

```
data={"phonetics":'T_UW1_TH|P_EY2_S_T', "rID":'487c3fe1-5f17-4010-a019-92b1c6ebfc5a'}
```
answer:
```
 b'{"status": "success", "stress_intensities": [[77, 33]], "stress_binaries": [[1, 0]]}'
```

### Lower level functions
Besides existing module, I am working on two lower level functions. 

The logic behind them is to use text and audio as input. The text is automatically phonetized and "grammarized", then htk model is used and:

- phonemeContrast gives you a detected transcription based on a target phoneme and a set of alternatives (in CMU phonemes)
```
url='/phonemeContrast'
data={"phonetics":phonetics, 'rID':rID, 'word_id':word_id, 'alternatives':alternatives, 'target':target}
```
As before, the phonetics is consituted of CMU phonemes with seperators for phonemes ("_"), syllables ("|") and words (" ").
For alternatives, one alternative is considered a word of several phonemes.

An example for a recording containing "I visited Italy". We want to study the phoneme "IH0" in "visited". I took this example because there are two of them:
```
data={'phonetics': 'AY1 V_IH1|Z_IH0|T_IH0_D IH1|T_AH0|L_IY0',
'rID': '487c3fe1-5f17-4010-a019-92b1c6ebfc5a',
'word_id': 1,
'alternatives': 'IH0 IY0',
'target': 'IH0'}
```

To study e.d. the "-ed" termination, you would need to input `'target': 'IH0_D'` and e.g. `'alternatives': "T D IH0_D"`.

And as there are two "IH0", I put alternatives for both, and return the detection of both like this:
```
b'{"status": "success", "phonetic_detection": ["IH0", "IH0"]}'
```

Now, for this same sentence "I visited Italy", we want to study the -ed termination of "visited", it would be like this:

The input:
```
data={'phonetics': '[["AY1"], ["V", "IH1", "Z", "IH0", "T", "IH0", "D"], ["IH1", "T", "AH0", "L", "IY0"]]',
'rID': b'487c3fe1-5f17-4010-a019-92b1c6ebfc5a',
'word_id': 1,
'alternatives': "['IH0 D', 'D', 'T']",
'target': 'IH0 D'}
```

and the output (if pronounced correctly):
```
b'{"status": "success", "phonetic_detection": ["IH0 D"]}'
```

- vowelStresses gives stress scores for each syllable of each word between 0 and 1
```
url='/vowel_stresses'
data={"text":text, 'filename':filename}
```
Example of output:
```
b'{"status": "success", "result": [[83], [49], [85], [27], [57], [27], [73, 44, 62]]}'
```

if you pass only one word with only one syllable, the result will be a `[[nan]]`


### Prefill feature for linguistic database

A prefill feature is available to generate phonetic content programatically. You need to make a POST request to `/prefill_from_phrase`, submitting a payload of type `application/x-www-form-urlencoded` (not JSON), for example:

```
url='/prefill_from_phrase'
data={"phrase":phrase}
```

Example:
Input:
```
data={"phrase":"Kayla isn't angry at Tyler"}
```

Output:

<!-- b'{"text": "I\'m taking a Spanish class.", 
"cmu_phonetics": "AY1_M T_EY1|K_IH0_NG AH0 S_P_AE1|N_IH0_SH K_L_AE1_S", 
"pronounciation_guide": "ay_m t_ey|k_i_ng uh s_p_a|n_i_sh k_l_a_s", 
"pronounciation_guide_hr": "aym tey|king uh spa|nish klas", 
"syllable_parts": "I\'m tak|ing a Span|ish class.", 
"n_syl_mismatch": 0, 
"used_method_for_syl_text": ["1 syl in cmu", "dataset", "1 syl in cmu", "dataset", "1 syl in cmu"]}' -->
```
b'{"text": "Kayla isn\'t angry at Tyler", 
"cmu_phonetics": "K_EY1|L_AH0 IH1|Z_AH0_N_T AE1_NG|G_R_IY0 AE1_T T_AY1|L_ER0", 
"pronounciation_guide": "k_ey|l_uh i|z_uh_n_t a_ng|g_r_ee a_t t_ay|l_uhr", 
"pronounciation_guide_hr": "key|luh i|zuhnt ang|gree at tay|luhr", 
"syllable_parts": "Kay|la isn\'t an|gry at Tyl|er",
"n_syl_mismatches": [0, 1, 0, 0, 0], 
"used_method_for_syl_text": ["SonoriPy", "dataset", "dataset", "1 syl in cmu", "dataset"], 
"cmu_phonetics_alt": [["K_EY1|L_AH0"], ["IH1|Z_AH0_N_T", "IH0|Z_AH0_N_T", "IH1|Z_AH0_N"], ["AE1_NG|G_R_IY0"], ["AE1_T"], ["T_AY1|L_ER0"]], 
"pronounciation_guide_alt": [["k_ey|l_uh"], ["i|z_uh_n_t", "i|z_uh_n_t", "i|z_uh_n"], ["a_ng|g_r_ee"], ["a_t"], ["t_ay|l_uhr"]], 
"pronounciation_guide_hr_alt": [["key|luh"], ["i|zuhnt", "i|zuhnt", "i|zuhn"], ["ang|gree"], ["at"], ["tay|luhr"]], 
"n_alternatives": [1, 3, 1, 1, 1]}'
```

## Docker application

You can also build the Dockerfile that will install everything and serve the application with Flask with nginx backend.
I used this info to do that: 
https://github.com/srcecde/flask-docker-ec2
<!-- https://github.com/ram-ch/Building-microservices-with-docker-on-AWS -->

First, clone this repo, then in it:

```
sudo docker-compose build
sudo docker-compose up -d
```

To kill all containers, e.g. to restart afterwards:
```
docker container kill $(docker ps -q)
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

(Not sure this is necessary)
change the line of nginx/web.conf
"	proxy_pass  http://aws.server.ip.here:5000/;"


If you just want to use it locally, without nginx server, you can build only flowspeech image:
```
docker build -t flowspeech .
docker run -d -p 8000:8000 flowspeech
```

When you want to update the app with changes in code (no new depencies):
```
git pull && docker-compose up -d --no-deps --build flaskapp
```

To git pull inside a container, this can be useful if you only change e.g. html and not python code (it does not rebuild the app, therefore faster):
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
