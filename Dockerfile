FROM mambaorg/micromamba
ARG DEBIAN_FRONTEND=noninteractive

USER root
# packages list. The --no-install-recommends avoids installing recommended packages that are not necessary for a tiny docker image: https://phoenixnap.com/kb/docker-image-size
RUN	apt-get update && apt-get install --no-install-recommends -y \
    libc6-dev-i386 \
    libx11-dev \
    gawk \
    curl \
    sudo \
    git \
	 make \
    cron \
    # These two are  necessary for pyworld library (f0 extraction)
    g++ \
    libsndfile-dev \
    # this is for git cloning  
    git-lfs \ 
    # dependencies for phonemizer
    festival espeak-ng \
    # clean up apt cache to save space
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install


# https://dev.to/emmanuelnk/using-sudo-without-password-prompt-as-non-root-docker-user-52bg
#  Add new user docker to sudo group
RUN adduser mambauser sudo
# Ensure sudo group users are not 
# asked for a password when using 
# sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
/etc/sudoers

USER mambauser
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
WORKDIR $HOME


# micromamba install was stuck, I applied this:
# https://stackoverflow.com/questions/76778360/micromamba-install-gets-stuck-when-run-in-docker-container-on-arm-mac
RUN micromamba config set extract_threads 1 # <---- This is the added line that fixes it

COPY --chown=$MAMBA_USER:$MAMBA_USER env_mfa_base.yml /tmp/env_mfa_base.yml
RUN micromamba install -y -n base -f /tmp/env_mfa_base.yml && \
    micromamba clean --all --yes

# https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
# Intall MFA from source (latest release), add a cpuonly in the yml just before pytorch dependency
RUN git clone https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner && \
    cd Montreal-Forced-Aligner && \
    # git checkout $(git describe --tags $(git rev-list --tags --max-count=1)) && \
    git checkout v2.2.15 && \
    pip install . && pip cache purge


COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes && pip cache purge

RUN echo "import nltk;nltk.download('averaged_perceptron_tagger')" | python
RUN echo "from transformers import Wav2Vec2Processor;processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')" | python
RUN mkdir /home/mambauser/hf_models && curl https://flwc-public-assets.s3.fr-par.scw.cloud/speech-models_last_hidden_state.quant.onnx -o /home/mambauser/hf_models/last_hidden_state.quant.onnx

RUN mkdir -p /home/mambauser/mfa
ENV MFA_ROOT_DIR=/home/mambauser/mfa
RUN mfa model download g2p french_mfa && mfa model download g2p spanish_spain_mfa && mfa model download g2p spanish_latin_america_mfa && mfa model download g2p english_uk_mfa && mfa model download g2p english_us_mfa  

# COPY --chown=$MAMBA_USER:$MAMBA_USER scripts/download_models.py download_models.py
# RUN mkdir /home/mambauser/hf_models/facebook && echo "import download_models" | python

WORKDIR /home/mambauser/code
CMD ["bash", "run_server.sh"]
EXPOSE 8000