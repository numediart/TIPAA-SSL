# Installation

## System dependencies

Install the packages listed in the [Dockerfile](./Dockerfile):

```bash
sudo apt-get install git-lfs espeak-ng festival
git lfs install
```

## Install micromamba

If you don't have micromamba yet, install it first:
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

## Create the environment

```bash
micromamba create -n flowspeech_mm -f env.yml
micromamba activate flowspeech_mm
pip install -e .
```

This installs all dependencies, including the ones needed to run the API and the tests.

## Notes for MacOS and Apple M1/... chips

A few adjustments are needed to be able to run on Apple Silicon, as some of the dependencies
on conda are not available for ARM architectures.

Install basic dependencies:

```bash
brew install git-lfs micromamba
```

Install homebrew for x86 and install dependencies:

```bash
arch -x86_64 zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install espeak
exit
```

Setup the conda environment for x86:

```bash
CONDA_SUBDIR=osx-64 micromamba create -n flowspeech_mm -f env.yml
micromamba activate flowspeech_mm
conda env config vars set PHONEMIZER_ESPEAK_PATH=/usr/local/bin/espeak
conda env config vars set PHONEMIZER_ESPEAK_LIBRARY=/usr/local/lib/libespeak.dylib
```

## Next steps

- Download the pretrained models: see [MODELS.md](./MODELS.md).
- Download the speech datasets: see [DATASETS.md](./DATASETS.md).
- Run the tests: see [TESTING.md](./TESTING.md).