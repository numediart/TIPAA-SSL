FROM continuumio/miniconda3
ARG DEBIAN_FRONTEND=noninteractive
# working directory
ENV HOME /root
WORKDIR $HOME

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

# https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
RUN mkdir -p /mfa
RUN conda create -p /env -c conda-forge "montreal-forced-aligner>=2.2"

# Python packages from conda
# ffmpeg is necessary to read mp3 files
RUN . activate /env && conda install ffmpeg && conda install python=3.10 && \
   # For using e.g. MelGAN or wav2vec2
   conda install pytorch torchaudio cpuonly -c pytorch && \
   # clean unnecessary setup files 
   conda clean --all -y


COPY ./requirements.txt $HOME/requirements.txt
# pip
RUN pip install --upgrade pip && pip install pyworld==0.3.2 && pip install -r requirements.txt

RUN echo "import nltk;nltk.download('averaged_perceptron_tagger')" | python

# As MFA cannot be ran from root, we have to create a new user and give him access to relevant folders
# https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
RUN useradd -ms /bin/bash mfauser
RUN chown -R mfauser /mfa
RUN chown -R mfauser /env
RUN chown -R mfauser /root
RUN chown -R mfauser /opt
USER mfauser
ENV MFA_ROOT_DIR=/mfa

RUN echo "source activate /env && mfa server start" > ~/.bashrc
ENV PATH /env/bin:$PATH

RUN . activate /env && mfa server init

# USER root
CMD ["bash", "run_server.sh"]
EXPOSE 8000