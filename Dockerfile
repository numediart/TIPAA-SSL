FROM mambaorg/micromamba
ARG DEBIAN_FRONTEND=noninteractive

USER root
# packages list. The --no-install-recommends avoids installing recommended packages that are not necessary for a tiny docker image: https://phoenixnap.com/kb/docker-image-size
RUN	apt-get update && apt-get install --no-install-recommends -y \
    libc6-dev-i386 \
    libx11-dev \
    gawk \
    curl \
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





USER mambauser

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

COPY --chown=$MAMBA_USER:$MAMBA_USER env_mfa_base.yml /tmp/env_mfa_base.yml

WORKDIR $HOME
# Intall MFA from source (latest release), add a cpuonly in the yml just before pytorch dependency
RUN git clone https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner && \
    cd Montreal-Forced-Aligner && \
    git checkout $(git describe --tags $(git rev-list --tags --max-count=1)) && \
    micromamba install -y -n base -f /tmp/env_mfa_base.yml && \
    micromamba clean --all --yes && \
    pip install .

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

RUN echo "import nltk;nltk.download('averaged_perceptron_tagger')" | python
RUN echo "from transformers import Wav2Vec2Processor;processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')" | python

# USER 
# https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
RUN mkdir -p /home/mambauser/mfa
ENV MFA_ROOT_DIR=/home/mambauser/mfa
RUN mfa model download g2p french_mfa && mfa model download g2p spanish_spain_mfa && mfa model download g2p spanish_latin_america_mfa && mfa model download g2p english_uk_mfa && mfa model download g2p english_us_mfa  

WORKDIR /home/mambauser/code
CMD ["bash", "run_server.sh"]
EXPOSE 8000