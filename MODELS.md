# Models

The pipeline builds on multilingual wav2vec 2.0 / XLS-R representations. The main pretrained representations used by the research are:

- `facebook/wav2vec2-large-xlsr-53`
- `facebook/wav2vec2-xlsr-53-espeak-cv-ft`

The latter was trained for multilingual phoneme recognition. In TIPAA-SSL, its learned representation is reused as a phonetic representation rather than simply taking its final phoneme sequence prediction.

## Download the base models

The [scripts/download_models.py](./scripts/download_models.py) script clones the Hugging Face repositories and pulls the LFS files into `hf_models/`:

```bash
python scripts/download_models.py
```

## Quantized ONNX model

The API and tests load the model in ONNX format from `hf_models/last_hidden_state.quant.onnx`. This is a compressed (quantized) version of the `wav2vec2-xlsr-53-espeak-cv-ft` model.

You can either:

- take a prebuilt `last_hidden_state.quant.onnx` (e.g. from the Flowchase drive folder) and place it at `hf_models/last_hidden_state.quant.onnx`, or
- build it yourself from the downloaded model with [scripts/onnx_utils.py](./scripts/onnx_utils.py):

  ```bash
  python -c \
    "from scripts.onnx_utils import convert_lhs_model_to_quant_onnx; \
     convert_lhs_model_to_quant_onnx('hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft')"
  ```

  This exports the last-hidden-state model to ONNX and quantizes it, writing `hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft/last_hidden_state.quant.onnx`. Move it to `hf_models/last_hidden_state.quant.onnx`.

## Downstream components

The dimensionality-reduction and frame-classifier components that run on top of the base wav2vec2 model are stored in this repository under [models/](./models).