#
# HTK (Hidden Markov Model Toolkit) Docker
# v3.4.1
# @author Loreto Parisi (loretoparisi at gmail dot com)
# v1.0.0
#
# Copyright (c) 2017 Loreto Parisi - https://github.com/loretoparisi/docker
#

# Modified by noe tits

#FROM frolvlad/alpine-miniconda3
FROM continuumio/miniconda3

# working directory
ENV HOME /root
WORKDIR $HOME


# Update repository list
RUN apt-get update

# packages list
RUN \
	apt-get update && apt-get install -y \
    libc6-dev-i386 \
    libx11-dev \
    gawk \
    curl \
    git \
	make

# pip
RUN pip install --upgrade pip

# Python packages from conda
RUN conda install -c anaconda -y python=3

# This is necessary so that librosa is able to read mp3 files
RUN conda install -c conda-forge ffmpeg

# This is necessary for pyworld library (f0 extraction)
RUN apt-get install -y g++

RUN apt-get install -y libsndfile-dev

ARG DEBIAN_FRONTEND=noninteractive

COPY ./ $HOME/
RUN pip install -r requirements.txt

# Install htk
# remove if exists, in case it is a shortcut from git
RUN rm -rf $HOME/htk/
RUN git clone https://github.com/loretoparisi/htk
WORKDIR $HOME/htk/
RUN ./configure --disable-hslab && \
    make all && \
    make install

# CMD ["bash"]
WORKDIR $HOME/
# following: https://runnable.com/docker/python/dockerize-your-flask-application
# ENTRYPOINT [ "python" ]
# CMD [ "/root/run_server.sh" ]

# https://gist.github.com/pangyuteng/f5b00fe63ac31a27be00c56996197597
# ENTRYPOINT ["python", "flask_server.py"]
# ENTRYPOINT ["gunicorn", "flask_server:app"]
# CMD ["gunicorn"  , "-b", "0.0.0.0:8000", "flask_server:app"]
EXPOSE 8000