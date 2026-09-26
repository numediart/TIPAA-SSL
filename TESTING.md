# Testing

The unit tests should work after the [installation](./INSTALLATION.md) steps, provided you have also:

- the quantized ONNX model in `hf_models/` (see [MODELS.md](./MODELS.md)), and
- the test data (see [DATASETS.md](./DATASETS.md)).

## Run the tests

```bash
pytest tests
```

The tests are grouped in three files:

- `tests/test_DL_modules.py` — pronunciation-analysis functions (stress, phonetic contrasts, text prefill)
- `tests/test_DL_api.py` — API endpoints (via Flask's test client)
- `tests/test_models.py` — the wav2vec2 frame-prediction model and the MFA forced-alignment test

## MFA models

`test_mfa_align` runs the Montreal Forced Aligner. It needs the `english_us_arpa` acoustic model and dictionary, which you can download once with:

```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

See also [the Dockerfile](./Dockerfile), which downloads these (plus g2p models) when building the image.