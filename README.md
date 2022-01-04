# Flowspeech

Project to migrate Flowchase speech tech from octave to python

## Web service

A Flask API is provided to access the modules.

Examples of client requests are available in `dummy_client.py`.

The process for processing speech has to be done in 2 separate requests:

1. Upload the audio file and get a request id (rID)
2. Call the right module (or a lower level function) with then needed params plus the rID

**To learn more about making requests to the Speech Processing API**, please see [this documentation file](/API.md).

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

## Process for updating code on server
- Do a pytest locally without container
- docker compose build containers locally
- docker compose up it locally, to check that it runs, and use dummy client on localhost
- On EC2: ssh to it, then, git pull and build flaskapp
- On ECS: docker context use myecs, then docker compose push (see below), then docker compose up (then docker context use default)
- or docker push on scaleway and see with Filipe

## Docker application on EC2

NOTE: To have debug mode on EC2, change in run_server.sh the line gunicorn to the commented "python flask_server.py"

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
ssh -i "~/flowspeech.pem" ec2-user@ec2-13-37-107-52.eu-west-3.compute.amazonaws.com
ssh -i "/mnt/c/Users/noe_t/Dropbox/contracts_info/flowchase/flowspeech.pem" ec2-user@ec2-13-37-107-52.eu-west-3.compute.amazonaws.com
```
You may have to `chmod 600 /mnt/c/Users/noe_t/Dropbox/contracts_info/flowchase/flowspeech.pem`
Under WSL, I had problems for changing these permissions, and followed the accepted answer here: https://superuser.com/questions/1323645/unable-to-change-file-permissions-on-ubuntu-bash-for-windows-10

But I had to install docker-compose like this:
```
pip install docker-compose
```

(Not necessary it seems, as well as the IP in the docker-compose.yml, it can be 0.0.0.0 as well)
change the line of nginx/web.conf
"	proxy_pass  http://aws.server.ip.here:8000/;"


If you just want to use it locally, without nginx server, you can build only flowspeech image:
```
docker build -t flowspeech .
docker run -d -p 8000:8000 flowspeech
```

When you want to update the app with changes in code (no new depencies):
```
docker kill flaskapp
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

To show terminal output:
```
docker logs flaskapp
```

## Docker application on ECR and ECS

In this setup, the idea is to docker push an image to a Amazons Elastic Container Registry (ECR), that will be used in the ECS.

How to setup a service (this is aweful, manual, and not working properly)
https://towardsdatascience.com/deploying-a-docker-container-with-ecs-and-fargate-7b0cbc9cd608
https://itnext.io/run-your-containers-on-aws-fargate-c2d4f6a47fda


Fortunately, with docker and aws-cli, there are interesting stuff:

https://devops4solutions.com/deploy-docker-container-in-ecs-using-docker-compose/
https://dev.to/sfrancavilla/deploy-web-apps-nginx-to-ecs-with-docker-198i

First create repos
```
aws ecr create-repository --repository-name speech_api
aws ecr create-repository --repository-name nginx
```

Connect to ECR
```
aws ecr get-login-password \
    --region eu-west-3 \
| docker login \
    --username AWS \
    --password-stdin 937215464284.dkr.ecr.eu-west-3.amazonaws.com
```

:warning: **In run_server.sh, make sure it is the gunicorn command to have the production server**
<!-- Push images to ECR:
```
docker push 937215464284.dkr.ecr.eu-west-3.amazonaws.com/speech_api
```
or with docker-compose, the yml has to conatain "image: ECR_URL":
`docker-compose push`

For only one of them, e.g.:
 docker-compose push nginx -->

The procedure to push and run a docker compose, or update it:
```
docker context use default
docker-compose -f docker-compose_aws.yml build
docker-compose -f docker-compose_aws.yml push
```
then:
```
docker context use myecs
docker compose -f docker-compose_aws.yml up
```
It takes time to update, but a new task is first created and afterwards the old one is removed.

```
aws ecs list-clusters
aws ecs list-tasks --cluster arn:aws:ecs:eu-west-3:937215464284:cluster/flowspeech
```

https://www.pulumi.com/docs/tutorials/aws/aws-py-fargate/

## Scaleway

You can login via `docker login rg.fr-par.scw.cloud/flowspeech -u nologin -p $SCW_SECRET_TOKEN`

After a local `docker compose build`,
the usual flow seems to be `docker tag flowspeech_flaskapp:latest rg.fr-par.scw.cloud/flowspeech/flowspeech_flaskapp:latest` + `docker push rg.fr-par.scw.cloud/flowspeech/flowspeech_flaskapp:latest`

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

### User recordings:

In my case, I have a .aws folder in my $HOME, with a config and credentials file (among others). 

I know you can have multiple profiles, so that you can work with different AWS regions and credentials. If you know how to set that up, great. I'm going to show you how my CLI is configured, that's what you'd need to grab the audio files. (edited)

My config (only the relevant bits):

```
...

[default]
region = fr-par
s3 =
    endpoint_url = https://s3.fr-par.scw.cloud
    signature_version = s3v4
    max_concurrent_requests = 100
    max_queue_size = 1000
    multipart_threshold = 50MB
    multipart_chunksize = 10MB
s3api =
    endpoint_url = https://s3.fr-par.scw.cloud
```

credentials: ask Filipe if I don't have it in twist or 1password.

Which means you can now download ~20GB of user recordings by running aws s3 sync s3://flwc-user-recordings ., but that's quite a lot. I can also give you just a sample, e.g. just our most active users or something like that.

Then symlink this folder in data/

## Test modules

```
pytest
```

## Using AWS Polly

Get an access ID from AWS: https://www.youtube.com/watch?v=VtuXrzfGLKQ&ab_channel=ActivePresenter

Doc:
https://docs.aws.amazon.com/polly/latest/dg/get-started-cli-exercise.html

The script ```tts_aws.py``` allows you to synthesize a list of sentences.

Use scp command to download them:

```
scp -r -i "~/flowspeech.pem" ec2-user@ec2-13-37-107-52.eu-west-3.compute.amazonaws.com:~/Flowspeech/synth_audio/ .
```

Details on AWS Polly CLI arguments and options:
https://docs.aws.amazon.com/cli/latest/reference/polly/synthesize-speech.html

SSML options:
https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html
