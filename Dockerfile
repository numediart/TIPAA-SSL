FROM mambaorg/micromamba:1.5.1-bookworm-slim
ARG DEBIAN_FRONTEND=noninteractive

USER root
# packages list. The --no-install-recommends avoids installing recommended packages that are not necessary for a tiny docker image: https://phoenixnap.com/kb/docker-image-size
RUN	apt-get update && apt-get install --fix-missing --no-install-recommends -y \
    gawk \
    curl \
    sudo \
    git \
    # this is for git cloning  
    git-lfs \ 
    # dependencies for phonemizer
    festival \
    espeak-ng \
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
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR $HOME

RUN micromamba config set extract_threads 1

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml \
    # need to do this until these guys fix their requirements
    && pip install --no-deps cmudict \
    && micromamba clean --all --yes && pip cache purge

RUN python -c "import nltk;nltk.download('averaged_perceptron_tagger')"
RUN python -c "from transformers import Wav2Vec2Processor;processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')"
RUN mkdir /home/mambauser/hf_models && \
    curl https://flwc-public-assets.s3.fr-par.scw.cloud/speech-models_last_hidden_state.quant.onnx -o /home/mambauser/hf_models/last_hidden_state.quant.onnx

RUN mkdir -p /home/mambauser/mfa
ENV MFA_ROOT_DIR=/home/mambauser/mfa
RUN mfa model download acoustic english_us_arpa &&\
    mfa model download dictionary english_us_arpa &&\
    mfa model download g2p french_mfa && mfa model download g2p spanish_spain_mfa &&\
    mfa model download g2p spanish_latin_america_mfa && \
    mfa model download g2p english_uk_mfa && mfa model download g2p english_us_mfa  

WORKDIR /home/mambauser/code

CMD ["bash", "run_server.sh"]
EXPOSE 8000
