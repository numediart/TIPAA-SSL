#
# HTK (Hidden Markov Model Toolkit) Docker
# v3.4.1
# @author Loreto Parisi (loretoparisi at gmail dot com)
# v1.0.0
#
# Copyright (c) 2017 Loreto Parisi - https://github.com/loretoparisi/docker
#

# Modified by noe tits

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
    # clean up apt cache to save space
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Python packages from conda
# This is necessary so that librosa is able to read mp3 files (in 2 steps to avoid conda memory error...)
# RUN conda install -c anaconda -y python=3 && conda install -c conda-forge nettle ffmpeg
# RUN conda install -c conda-forge nettle ffmpeg && \
RUN conda install ffmpeg && \
# For using e.g. MelGAN or wav2vec2
   conda install pytorch torchaudio cpuonly -c pytorch && \
# clean unnecessary setup files 
   conda clean --all -y

COPY ./requirements.txt $HOME/requirements.txt
# COPY ./ $HOME/

# pip
RUN pip install --upgrade pip && pip install pyworld && pip install -r requirements.txt

# load melgan model now so that it is cached later
# RUN echo "import torch;torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')" | python

# Install htk
# remove if exists, in case it is a shortcut from git that was copied above
RUN rm -rf $HOME/htk/ &&\
    git clone https://github.com/loretoparisi/htk &&\
    cd $HOME/htk/ && ./configure --disable-hslab && \
    make all && \
    make install


# from https://stackoverflow.com/questions/37458287/how-to-run-a-cron-job-inside-a-docker-container
# copy crontabs for root user
COPY cronjobs/remove_old_files /etc/crontabs/root
RUN crontab /etc/crontabs/root
CMD ["bash", "run_server.sh"]
EXPOSE 8000