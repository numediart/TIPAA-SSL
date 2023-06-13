## Information on tech implementation

This is the model of xlsr wav2vec2 paper (no fine-tuning):
https://huggingface.co/facebook/wav2vec2-large-xlsr-53

This model is the xlsr 53 that was then fintuned towards prediction of phoneme sequences (the phoneme sequences come from the transcriptions that were phonetized with the help of espeak tool, which is thus a g2p in IPA, but I would say a not very standardized IPA, and also not compatible with a series of tools like MFA)
https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft

This class wraps the different steps of the model pipeline:
https://github.com/flowchase/flowspeech/blob/63500407041dbc953fcfac02f69847abf072066a/src/wav2vec2_frame_prediction.py#L40

It uses the the xlsr finetuned (let' call it xlsr-53-ft), but the idea is to not use the phonemes labels predicted by the model, but the latent representation that is at the last layer that represent a phonetic space from which we can derive different things.

For this we train another model on a task of frame-level classifier (very light model) where 1 frame is a latent vector from xlsr-53-ft. Let's call this "latent frame".

To have the data to train this classifier, we need to build a dataset of frames along with their associated phoneme label.

For this we use MFA to align a speech dataset with its IPA phonetic standard.
Explanations of their choices on IPA:
https://mfa-models.readthedocs.io/en/latest/mfa_phone_set.html

MFA has:
-g2p models: for missing words in the dictionary, the g2p models predict the list of phonemes
-pronunciation dictionaries, i.e. list of words with associated lists of phonemes
-machinel learning models that align a pre-determined list of phonemes on an audio file

What we do with MFA is thus:

extract phonetics from text (in their IPA standard, or in CMU if used only in english)
run a forced alignment HMM model that predict the start and end timings of every phoneme (on the set of sentences)
With this information, we can then pass the audio in the xlsr-53-ft to get the latent frames, and thanks to the timings, we know what phonemes are associated to every latent frames.

This allows us to train the very light frame-level classifier. (In fact I first do a dimension reduction of the latent frames, and after that I train the reduced latent frame classifier)

This model pipeline can thus predict from an audio, a probability matrix. This matrix is a sequence of probability vectors corresponding to the latent frames. (Let's call that probability frames or phoneme probability frames)

This pipeline is implemented in a class that gather the different models
https://github.com/flowchase/flowspeech/blob/main/src/wav2vec2_frame_prediction.py#L40

The fit function trains the different steps in series:
https://github.com/flowchase/flowspeech/blob/main/src/wav2vec2_frame_prediction.py#L110

The function that is used to predict this phoneme probability matrix:
https://github.com/flowchase/flowspeech/blob/main/src/wav2vec2_frame_prediction.py#L124

There is a last step that does not depend on a trained machine learning model, but on a well-known algorithm: Dynamic Time Warping.

In this function, I first call the latter function prediction the probability matrx, and then I use the "target_phonemes" which here represent the list of phonemes corresponding to what is expected (the content of the exercise that the learner is supposed to say)
https://github.com/flowchase/flowspeech/blob/main/src/wav2vec2_frame_prediction.py#L155

As you can see, it calls a function from another class I did, dedicated to the DTW part:
https://github.com/flowchase/flowspeech/blob/main/src/dtw_forced_aligner.py#L187

Inside that, the function that actually performs DTW by calling a function from librosa:
https://github.com/flowchase/flowspeech/blob/main/src/dtw_forced_aligner.py#L49

The "df_segmented" variable outputted by the "predict_with_timings" function in "wav2vec2_frame_prediction" is a dataframe that contains the timings of the expected phonemes, and the classification result for each of these expecting phoneme as well as more detailed probability results.

In here, you can see different models I tried to train based on pickels containing sets of frames that are pre-computed else where (these are not on the github because it would be a mess)
https://github.com/flowchase/flowspeech/blob/main/src/wav2vec2_frame_prediction.py#L162

Before choosing models that I thought were working well, I experimented things at the frame level only in here:
https://github.com/flowchase/flowspeech/blob/main/scripts/frame_classifiers_experiments.py

## Theory on the method

This paper "SIMPLE AND EFFECTIVE ZERO-SHOT CROSS-LINGUAL PHONEME RECOGNITION" (https://arxiv.org/pdf/2109.11680.pdf) corresponds to the basis model I use: https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft

This model is trained in 2 phases

self-supervised training on 53 languages (with a kind of reconstruction task, based on a contrastive loss) (the result is this model )
then it is trained on a phoneme sequence prediction, also on a multilingual dataset, with a CTC loss. The phoneme labels for this training phase are obtained thanks to “phonemizer” library that uses “espeak” backend that is able to perform “grapheme-to-phoneme” in a lot of languages. However it is not “standardized” enough (as I explain in the scientific report)
Therefore, on top of this

I use the former model to extract last hidden vectors. I can access them at frame level, and therefore have full control over the timings. We can enjoy the performance of learned representations without having the constraints of the CTC collapsing to output a phoneme sequence
Also, by starting from this latent representation, we can train new models that will have different phoneme sets, we can use different sets of data with different accents, languages and experiment with that
For now I thus trained a frame-level classifier, preceded by a reducer, with any dataset we want with any label we want. By experience I found out that that methodology allows to use shallow model for reduction and classification (because a lot of the complex relationships are modeled thanks to wav2vec2 and mapped to a good phonetic space thanks to the training towards phoneme sequences with CTC
DTW is used between expected phonetics and the frame-level output probabilities. This allows to do classification on a target phoneme. This is the task we need to give the feedback “You said this phoneme instead of that phoneme” (edited)