# Flowspeech

Flowchase's speech tech

## Web service

A Flask API is provided to access the speech analysis features by pronunciation aspect.
**To learn more about making requests to the Speech Processing API**, please see [this documentation file](/API_v2.md).

## Process for updating code on server
- Do a pytest locally without container
- docker compose build containers locally
- docker compose up it locally, to check that it runs, and use test_DL_api.py on localhost
- On web server: ssh to it, then, git pull and build flaskapp

## Docker application

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

<details>
<summary>Try torch compile (optional)</summary>
<br>
For torch.compile() optimizations, see https://pytorch.org/get-started/pytorch-2.0/#getting-started :
In CPU:
`pip3 install numpy numba --pre torch torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

GPU:
`pip3 install numpy --pre torch[dynamo] torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu116`


For GPU support on Windows with WSL2, I followed this:
https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl
https://github.com/pytorch/pytorch/issues/73487#issuecomment-1115441977

i.e.: `pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html`

Then for doing AutoML: `pip install autoPyTorch`

</details>

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

I created a profile called `iam_user` with the two lines in `~/.aws/credentials`
```
[user1]
aws_access_key_id=...
aws_secret_access_key=...
```
Then used `aws configure --profile iam_user`.

see:
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html


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
