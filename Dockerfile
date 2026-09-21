FROM mambaorg/micromamba:1.5.1-bookworm-slim

ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update && apt-get install --fix-missing --no-install-recommends -y \
    gawk \
    curl \
    sudo \
    git \
    pipx \
    git-lfs \
    festival \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN adduser mambauser sudo

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER mambauser

RUN micromamba config set extract_threads 1

ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY --chown=$MAMBA_USER:$MAMBA_USER conda-lock.yml /tmp/conda-lock.yml

RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install -y -n base -f /tmp/conda-lock.yml \
    && micromamba run -n base python -m pip install --no-cache-dir \
         "onnx==1.16.1" \
         pytest \
    && micromamba run -n base python -c \
         "import onnx, onnxruntime; print('onnx', onnx.__version__, 'onnxruntime', onnxruntime.__version__)"

RUN mkdir -p /home/mambauser/mfa

ENV MFA_ROOT_DIR=/home/mambauser/mfa

RUN mfa model download acoustic english_us_arpa && \
    mfa model download dictionary english_us_arpa && \
    mfa model download g2p english_uk_mfa && \
    mfa model download g2p english_us_mfa

RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

RUN python -c \
    "from transformers import Wav2Vec2Processor; \
     Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')"

WORKDIR /home/mambauser/code

COPY . .

RUN python -m pip install -e .

CMD ["/bin/bash", "run_server.sh"]