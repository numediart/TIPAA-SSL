# TIPAA-SSL & MUST&P-SRL

## Open-source implementations for multilingual speech representation, syllabification, and text-independent phonetic alignment

This repository contains the research implementations behind two publications developed during R&D on AI-based pronunciation analysis:

* **TIPAA-SSL** — *Text-Independent Phone-to-Audio Alignment Leveraging SSL (TIPAA-SSL) Pre-Trained Model Latent Representation and Knowledge Transfer*
* **MUST&P-SRL** — *Multi-lingual and Unified Syllabification in Text and Phonetic Domains for Speech Representation Learning*

The code combines self-supervised speech representations, phonetic processing, frame-level phoneme classification, forced alignment, dynamic time warping, and multilingual linguistic processing.

The original research was motivated by pronunciation analysis and language-learning applications, but the techniques are applicable more broadly to **speech analysis, phonetic alignment, pronunciation assessment, speech representation learning, and multilingual speech technology**.

---

## Research papers

### 1. TIPAA-SSL

**Text-Independent Phone-to-Audio Alignment Leveraging SSL (TIPAA-SSL) Pre-Trained Model Latent Representation and Knowledge Transfer**

Noé Tits, Prernna Bhatnagar, Thierry Dutoit
*Acoustics*, 2024, 6(3), 772–781.

* [Paper — MDPI](https://www.mdpi.com/2624-599X/6/3/42)
* [DOI](https://doi.org/10.3390/acoustics6030042)
* [Implementation](flowspeech/wav2vec2_frame_prediction.py)
* [Technical implementation notes](INFO.md)

### What is TIPAA-SSL?

TIPAA-SSL investigates how the latent representations of a multilingual self-supervised speech model can be used for **frame-level phonetic analysis and text-independent phone-to-audio alignment**.

Instead of directly using the phoneme sequence predicted by a CTC-based wav2vec 2.0 model, the approach extracts its intermediate/last-layer representations and uses them as a learned phonetic representation.

A lightweight downstream pipeline then performs:

1. self-supervised speech representation extraction;
2. dimensionality reduction;
3. frame-level phoneme classification;
4. phoneme probability estimation for each audio frame;
5. optional alignment against an expected phoneme sequence using Dynamic Time Warping (DTW).

This provides access to phonetic information while retaining explicit frame-level timing information.

---

## 2. MUST&P-SRL

**MUST&P-SRL: Multi-lingual and Unified Syllabification in Text and Phonetic Domains for Speech Representation Learning**

Noé Tits
*Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track*, pages 74–82.

* [Paper — ACL Anthology](https://aclanthology.org/2023.emnlp-industry.8/)
* [DOI](https://doi.org/10.18653/v1/2023.emnlp-industry.8)
* [Implementation](flowspeech/text_processing.py)
* Main function: `prefill_for_sentence`

### What is MUST&P-SRL?

MUST&P-SRL provides multilingual linguistic processing for extracting structured information from text and phonetic representations.

The approach includes:

* multilingual phonetic transcription;
* syllabification;
* stress information;
* unified representations in text and phonetic domains;
* processing compatible with speech alignment workflows;
* integration with open-source linguistic and speech-processing tools.

The method was evaluated on English, French, and Spanish data and was designed to support speech representation learning and downstream speech-analysis applications.

---

# Why these two pieces of research are together

The two publications address complementary parts of a speech-analysis pipeline.

**MUST&P-SRL** provides structured linguistic and phonetic representations, including syllable-level information.

**TIPAA-SSL** provides a way of mapping learned speech representations to frame-level phonetic information and, when an expected phonetic sequence is available, aligning that representation with the audio.

Together, they form part of a broader speech-processing stack for pronunciation analysis and related applications.

---

# System overview

The TIPAA-SSL pipeline can be summarized as:

```text
                         Audio
                           │
                           ▼
                 ┌────────────────────┐
                 │   wav2vec 2.0 /    │
                 │      XLS-R          │
                 └─────────┬──────────┘
                           │
                           ▼
                 Frame-level latent
                    representations
                           │
                           ▼
                 ┌────────────────────┐
                 │ Dimension          │
                 │ reduction          │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Frame-level        │
                 │ phoneme classifier │
                 └─────────┬──────────┘
                           │
                           ▼
                 Phoneme probability
                      per frame
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
             Frame-level      Expected
              phonetic        phoneme
              analysis         sequence
                                  │
                                  ▼
                         Dynamic Time Warping
                                  │
                                  ▼
                       Phone-to-audio alignment
```

The key idea is to use the learned phonetic space of a multilingual self-supervised model while retaining access to frame-level information instead of relying only on the final collapsed CTC phoneme sequence.

---

# Repository structure

The repository contains the implementation and supporting material for the speech-processing pipeline.

```text
.
├── flowspeech/              # Main speech and linguistic processing package
├── models/                  # Trained downstream model components
├── data/                    # Data and processing resources
├── scripts/                 # Offline processing and experiments
├── syllabipy/               # Modified syllabification component
├── content_tools/           # Linguistic/content-processing interface
├── app/                     # API application
├── nginx/                   # Deployment configuration
├── notebooks/               # Experimental notebooks
├── README.md                # This file
├── INFO.md                  # Technical implementation notes
├── API_v2.md                # API documentation
├── DEPLOYMENT.md            # Deployment information
├── INSTALLATION.md          # Installation instructions (incl. macOS)
├── MODELS.md                # Pretrained model download and setup
├── DATASETS.md              # Speech dataset download instructions
├── TESTING.md               # How to run the tests
└── DEVELOPMENT.md           # Best practices, notebooks and TODO
```

The most relevant implementation files for the two papers are:

* **TIPAA-SSL:** `flowspeech/wav2vec2_frame_prediction.py`
* **MUST&P-SRL:** `flowspeech/text_processing.py`

---

# Getting started

## Requirements

The project currently targets Python 3.10+.

Some components additionally require:

* Git LFS
* eSpeak / eSpeak NG
* Festival
* micromamba / conda-compatible environment management

The repository contains the downstream model components (dimensionality reduction and frame classifier) in `models/`. The base wav2vec2 models are downloaded separately, see [MODELS.md](MODELS.md).

## Clone the repository

```bash
git clone https://github.com/numediart/TIPAA-SSL.git
cd TIPAA-SSL
```

## Install dependencies

Install the system dependencies:

```bash
sudo apt-get install git-lfs espeak-ng festival
git lfs install
```

Then create the project environment:

```bash
micromamba create -n flowspeech_mm -f env.yml
micromamba activate flowspeech_mm
pip install -e .
```

For Apple Silicon / macOS, see the platform-specific instructions in [INSTALLATION.md](INSTALLATION.md).

See [INSTALLATION.md](INSTALLATION.md) for the full instructions, [MODELS.md](MODELS.md) for the pretrained models, [DATASETS.md](DATASETS.md) for the speech datasets, and [TESTING.md](TESTING.md) for how to run the tests.

---

# Models

The pipeline builds on multilingual wav2vec 2.0 / XLS-R representations.

The main pretrained representations used by the research are based on:

* `facebook/wav2vec2-large-xlsr-53`
* `facebook/wav2vec2-xlsr-53-espeak-cv-ft`

The latter model was trained for multilingual phoneme recognition. In TIPAA-SSL, its learned representation is reused as a phonetic representation rather than simply taking its final phoneme sequence prediction.

The downstream dimensionality-reduction and frame-classification components used by the pipeline are included in this repository under:

```text
models/
```

See [MODELS.md](MODELS.md) for how to download the base wav2vec2 models and set up the quantized ONNX model used by the API and tests.

---

# Data and forced alignment

Training the downstream frame-level classifier requires audio frames associated with phoneme labels.

For this purpose, the research pipeline uses forced alignment tools such as the **Montreal Forced Aligner (MFA)** to obtain phoneme-level timing information.

The resulting aligned data can then be associated with the frame-level representations extracted from the self-supervised speech model.

This makes it possible to train a lightweight classifier on top of the pretrained representation.

See [DATASETS.md](DATASETS.md) for instructions on downloading the speech datasets used for training and evaluation.

---

# Text-independent vs. text-conditioned alignment

One important distinction in this work is between:

### Frame-level phonetic analysis

The model can produce phoneme probability distributions for successive audio frames without requiring a transcript to be provided to the frame classifier.

### Alignment with an expected pronunciation

When an expected phoneme sequence is available, Dynamic Time Warping can be used to align the expected phonemes with the frame-level probability sequence.

This distinction is important for understanding the TIPAA-SSL method and its potential use in pronunciation analysis.

---

# Relation to pronunciation analysis

The research was originally developed in the context of AI-based pronunciation analysis and feedback.

A pronunciation-analysis system can use the resulting phonetic information to investigate questions such as:

* Which phoneme was produced?
* When does a phoneme occur in the audio?
* Was an expected phoneme omitted or substituted?
* How closely does the pronunciation follow an expected phonetic sequence?
* Which parts of a pronunciation require further analysis?

The released implementation is therefore useful not only as a pronunciation-training system, but also as a research platform for experimenting with **phonetic representations and alignment**.

---

# Documentation

| Document | Description |
|----------|-------------|
| [INFO.md](INFO.md) | Detailed explanation of the TIPAA-SSL implementation |
| [API_v2.md](API_v2.md) | API endpoints documentation |
| [INSTALLATION.md](INSTALLATION.md) | Installation instructions, incl. macOS/Apple Silicon |
| [MODELS.md](MODELS.md) | Pretrained model download and setup (`hf_models/`, quantized ONNX) |
| [DATASETS.md](DATASETS.md) | Speech dataset download instructions (actor recordings, LibriSpeech, MAILABS) |
| [TESTING.md](TESTING.md) | How to run the tests and what they need |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Best practices, Jupyter notebooks, TODO |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment information |

---

# Research talk

A presentation covering the two research contributions was given as part of the **ISCA SIG-SLATE webinar series**.

[Watch the presentation on YouTube](https://www.youtube.com/watch?v=7_pQ0aQwg-w)

---

# Citation

If you use this repository in academic work, please cite the relevant paper(s).

### TIPAA-SSL

```bibtex
@Article{acoustics6030042,
  author  = {Tits, Noé and Bhatnagar, Prernna and Dutoit, Thierry},
  title   = {Text-Independent Phone-to-Audio Alignment Leveraging SSL (TIPAA-SSL) Pre-Trained Model Latent Representation and Knowledge Transfer},
  journal = {Acoustics},
  volume  = {6},
  number  = {3},
  pages   = {772--781},
  year    = {2024},
  doi     = {10.3390/acoustics6030042}
}
```

### MUST&P-SRL

```bibtex
@inproceedings{tits-2023-must,
  title     = {MUST\&P-SRL: Multi-lingual and Unified Syllabification in Text and Phonetic Domains for Speech Representation Learning},
  author    = {Tits, Noé},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  pages     = {74--82},
  year      = {2023},
  doi       = {10.18653/v1/2023.emnlp-industry.8}
}
```

A software citation will also be provided through the repository's `CITATION.cff` and archived software release.

---

# License

The repository is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.

Please also consult the license and attribution information associated with individual third-party components included in or used by the project.

In particular, some dependencies and adapted components have their own licenses and attribution requirements.

---

# Status

This repository is a public research release of software developed for speech and language technology research.

Some parts of the original production system were developed for a larger application and may require adaptation before being used as a standalone research package.

The repository is therefore intended primarily as a **research and experimentation resource**, rather than as a polished production-ready speech API.

Contributions, experiments, issues, and discussion are welcome.

---

# Acknowledgements

This work was developed in the context of research and development at **Flowchase** and the **Numediart Institute of UMONS**.

The research builds on open-source speech and language technologies including wav2vec 2.0 / XLS-R, the Montreal Forced Aligner, eSpeak, and related open-source resources.
