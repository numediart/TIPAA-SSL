# Information on tech implementation

## Base models

This is the model of the XLSR wav2vec2 paper (no fine-tuning):
[facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)

This model is the XLSR-53 that was then finetuned towards prediction of phoneme sequences (the phoneme sequences come from the transcriptions that were phonetized with the help of the espeak tool, which is thus a g2p in IPA, but I would say a not very standardized IPA, and also not compatible with a series of tools like MFA):
[facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)

## Model pipeline

The class `Wav2Vec2ForFramePrediction` in [flowspeech/wav2vec2_frame_prediction.py](flowspeech/wav2vec2_frame_prediction.py) wraps the different steps of the model pipeline.

It uses the XLSR finetuned model (let's call it XLSR-53-ft), but the idea is to not use the phoneme labels predicted by the model, but the latent representation that is at the last layer, which represents a phonetic space from which we can derive different things.

For this we train another model on a task of frame-level classification (a very light model) where 1 frame is a latent vector from XLSR-53-ft. Let's call this a "latent frame".

To have the data to train this classifier, we need to build a dataset of frames along with their associated phoneme label.

## Forced alignment with MFA

For this we use the Montreal Forced Aligner (MFA) to align a speech dataset with its IPA phonetic standard.
Explanations of their choices on IPA:
[MFA phone set](https://mfa-models.readthedocs.io/en/latest/mfa_phone_set.html)

MFA has:

- g2p models: for missing words in the dictionary, the g2p models predict the list of phonemes
- pronunciation dictionaries, i.e. list of words with associated lists of phonemes
- machine learning models that align a pre-determined list of phonemes on an audio file

What we do with MFA is thus:

1. extract phonetics from text (in their IPA standard, or in CMU if used only in English)
2. run a forced alignment HMM model that predicts the start and end timings of every phoneme (on the set of sentences)
3. with this information, we can then pass the audio in the XLSR-53-ft to get the latent frames, and thanks to the timings, we know what phonemes are associated to every latent frame.

This allows us to train the very light frame-level classifier. (In fact I first do a dimension reduction of the latent frames, and after that I train the reduced latent frame classifier)

This model pipeline can thus predict from an audio a probability matrix. This matrix is a sequence of probability vectors corresponding to the latent frames. (Let's call that probability frames or phoneme probability frames)

## Main functions

- `Wav2Vec2ForFramePrediction.fit` trains the different steps in series.
- `Wav2Vec2ForFramePrediction.predict_phone_prob_matrix` is the function used to predict this phoneme probability matrix.
- There is a last step that does not depend on a trained machine learning model, but on a well-known algorithm: Dynamic Time Warping. In this function, I first call the latter function predicting the probability matrix, and then I use the "target_phonemes" which here represent the list of phonemes corresponding to what is expected (the content of the exercise that the learner is supposed to say). As you can see, it calls a function from another class I did, dedicated to the DTW part: [flowspeech/dtw_forced_aligner.py](flowspeech/dtw_forced_aligner.py).
- Inside that, the function that actually performs DTW by calling a function from librosa.

The `df_segmented` variable outputted by the `phone_prob_matrix_segmentation` function in `Wav2Vec2ForFramePrediction` is a dataframe that contains the timings of the expected phonemes, and the classification result for each of these expected phonemes as well as more detailed probability results.

In [flowspeech/wav2vec2_frame_prediction.py](flowspeech/wav2vec2_frame_prediction.py), you can also see different models I tried to train based on pickles containing sets of frames that are pre-computed elsewhere (these are not on the github because it would be a mess).

Before choosing models that I thought were working well, I experimented things at the frame level only in here: [scripts/frame_classifiers_experiments.py](scripts/frame_classifiers_experiments.py)

# Theory on the method

This paper "SIMPLE AND EFFECTIVE ZERO-SHOT CROSS-LINGUAL PHONEME RECOGNITION" (https://arxiv.org/pdf/2109.11680.pdf) corresponds to the basis model I use: [facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)

This model is trained in 2 phases:

1. self-supervised training on 53 languages (with a kind of reconstruction task, based on a contrastive loss) (the result is this model)
2. then it is trained on a phoneme sequence prediction, also on a multilingual dataset, with a CTC loss. The phoneme labels for this training phase are obtained thanks to the "phonemizer" library that uses the "espeak" backend that is able to perform "grapheme-to-phoneme" in a lot of languages. However it is not "standardized" enough (as I explain in the scientific report)

Therefore, on top of this:

- I use the former model to extract last hidden vectors. I can access them at frame level, and therefore have full control over the timings. We can enjoy the performance of learned representations without having the constraints of the CTC collapsing to output a phoneme sequence
- Also, by starting from this latent representation, we can train new models that will have different phoneme sets, we can use different sets of data with different accents, languages and experiment with that
- For now I thus trained a frame-level classifier, preceded by a reducer, with any dataset we want with any label we want. By experience I found out that this methodology allows to use shallow models for reduction and classification (because a lot of the complex relationships are modeled thanks to wav2vec2 and mapped to a good phonetic space thanks to the training towards phoneme sequences with CTC)
- DTW is used between expected phonetics and the frame-level output probabilities. This allows to do classification on a target phoneme. This is the task we need to give the feedback "You said this phoneme instead of that phoneme"