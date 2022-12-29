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
    g++ \
    libsndfile-dev \
    # this is for git cloning  
    git-lfs \ 
    # clean up apt cache to save space
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Python packages from conda
# This is necessary so that librosa is able to read mp3 files
RUN conda install ffmpeg && \
# For using e.g. MelGAN or wav2vec2
   conda install pytorch torchaudio cpuonly -c pytorch && \
#    conda install -c conda-forge tensorflow-cpu && \
   # conda install -c conda-forge montreal-forced-aligner && \
#    conda install tensorflow && \
# clean unnecessary setup files 
   conda clean --all -y


COPY ./requirements.txt $HOME/requirements.txt
# pip
RUN pip install --upgrade pip && \
      # pip install pyworld==0.3 && \
      pip install -r requirements.txt

CMD ["bash", "content_tools/run_content_tools.sh"]
# CMD ["streamlit", "run", "content_tools/Hello.py"]

EXPOSE 8501