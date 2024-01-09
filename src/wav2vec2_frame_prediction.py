import logging
import os
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from time import time
from typing import cast

import numpy as np
import pandas as pd
import torch
from linetimer import CodeTimer
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from strenum import StrEnum
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.audio_processing import (
    getIntensity,
    getIntonation,
    read_audio_bytes,
    read_audio_file,
    read_audio_string,
)
from src.dtw_forced_aligner import dtw_forced_aligner
from src.pronunciation_dictionaries import (
    cmu_alphabet,
    cmu_phones_info,
    cmu_stressed_alphabet,
    ipa_alphabet,
)
from src.text_processing import remove_grouping_hyphens

cmu_vowels = [p[0] for p in cmu_phones_info if p[1][0] == 'vowel']
cmu_consonants = [p[0] for p in cmu_phones_info if p[1][0] != 'vowel']

logger = logging.getLogger(__name__)

DEFAULT_MIN_SPEECH_RATE = 1  # syll/s
DEFAULT_MAX_SPEECH_RATE = 8  # syll/s
DEFAULT_SILENCE_THRESHOLD = 40  # dBFS


# StrEnum in python 3.11
class AudioStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "success, audio is empty (has zero sample)"
    TOO_SHORT = "success, audio is too short compared to the expected number of syllables"
    TOO_LONG = "success, audio is too long compared to the expected number of syllables"
    NO_SOUND = "success, no voiced sound detected (only 0's in waveform)"
    TOO_QUIET = "success, volume too low or no voice detected (intensity too low)"
    NO_PITCH = "success, no voiced sound detected (no pitch detected)"
    DTW_FAILED = "success, DTW alignment failed"
    PHONETIC_DETECTION_FAILED = "success, phonetic detection failed"
    PHONETIC_DETECTION_SILENCE = "success, phonetic detection failed (only silence)"
    NO_MATCH = "success, phone error rate too high"
    FILE_NOT_FOUND = "error, audio file not found"


@dataclass
class AudioLoadResult:
    """Result of loading audio, with status, waveform, sampling rate, and pitch frame ratio

    status: AudioStatus
        status of the file loading and checking
    waveform: np.ndarray | None
        waveform
    sampling_rate: float | None
        sampling rate in Hz
    speech_rate: float | None
        syllable frequency in syllables per second
    silent_sample_ratio: float | None
        ratio of silent frames
    pitch_sample_ratio: float | None
        ratio of frames with pitch detected
    """

    status: AudioStatus
    waveform: np.ndarray | None
    sampling_rate: float | None
    speech_rate: float | None
    silent_sample_ratio: float | None
    pitch_sample_ratio: float | None


class AudioMode(Enum):
    """Mode of audio input

    FILE: file path
    BASE64: base64 encoded string
    BYTES: raw bytes
    NUMPY: numpy array of waveform
    """

    FILE = 0
    BASE64 = 1
    BYTES = 2
    NUMPY = 3


# File path, or raw bytes, or base64 encoded string, or numpy array
# Depending on the mode (AudioMode)
AudioInput = str | bytes | bytearray | np.ndarray


def audio_load_and_check(
    audio: AudioInput,
    phonetics: str,
    min_speech_rate: float = DEFAULT_MIN_SPEECH_RATE,
    max_speech_rate: float = DEFAULT_MAX_SPEECH_RATE,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    mode: AudioMode = AudioMode.FILE,
    fs: float = 16000,
) -> AudioLoadResult:
    """Load audio with modes: from a "file", from "base64" encoding, from "bytes", or directly a "numpy" array
    Then check duration to see if it's plausible

    I first detect if the audio is too short to have a realistic speech rate
        https://www.science.org/doi/10.1126/sciadv.aaw2594
    https://www.reddit.com/r/languagelearning/comments/f5o1om/distribution_of_syllable_rate_sr_in_syllables_per/
    Speech rate is always between 5 and 8 syl/second
    """
    n_syllables_tot = sum([len(el.split('|')) for el in phonetics.split(' ')])

    if mode == AudioMode.FILE:
        if not isinstance(audio, str):
            raise TypeError(f"audio must be of type str, not {type(audio)}")
        try:
            s, fs = read_audio_file(audio, fs=fs)
        except FileNotFoundError:
            return AudioLoadResult(
                AudioStatus.FILE_NOT_FOUND, None, None, None, None, None
            )
    elif mode == AudioMode.BASE64:
        if isinstance(audio, np.ndarray):
            raise TypeError(f"audio must be of type str or bytes, not {type(audio)}")
        s, fs = read_audio_string(audio, fs=fs)
    elif mode == AudioMode.BYTES:
        if not isinstance(audio, bytes | bytearray):
            raise TypeError(f"audio must be of type bytes, not {type(audio)}")
        s, fs = read_audio_bytes(audio, fs=fs)
    elif mode == AudioMode.NUMPY:
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"audio must be of type np.ndarray, not {type(audio)}")
        s: np.ndarray = audio
    else:
        raise ValueError(f"Unknown audio mode {mode}")

    if len(s) == 0:
        return AudioLoadResult(AudioStatus.EMPTY, None, None, None, None, None)

    if np.abs(s).sum() == 0:
        return AudioLoadResult(AudioStatus.NO_SOUND, s, fs, None, None, None)

    # detect silent frames using intensity
    intensity = getIntensity(s, fs)
    silent_frames = intensity < silence_threshold
    silent_sample_ratio = silent_frames.sum() / len(s)

    if silent_frames.all():
        return AudioLoadResult(AudioStatus.TOO_QUIET, s, fs, None, None, None)

    # ideally we would detect the onset and the end of the
    # voiced part of the audio, and only compute the speech rate
    # on that part
    nonsilent_duration = (~silent_frames).sum() / fs
    speech_rate = n_syllables_tot / nonsilent_duration

    f0_samples = getIntonation(s, fs)
    # pitch frame ratio: ratio of nonsilent frames with pitch detected
    pitch_sample_ratio = ((f0_samples > 0) & ~silent_frames).sum() / (
        ~silent_frames
    ).sum()

    if speech_rate > max_speech_rate:
        status = AudioStatus.TOO_SHORT
    # too tricky to have a reliable speech rate for very short audios,
    # so don't check if only one syllable
    elif speech_rate < min_speech_rate and n_syllables_tot > 1:
        status = AudioStatus.TOO_LONG
    elif pitch_sample_ratio == 0:
        status = AudioStatus.NO_PITCH
    else:
        status = AudioStatus.SUCCESS

    return AudioLoadResult(
        status, s, fs, speech_rate, silent_sample_ratio, pitch_sample_ratio
    )


def extract_word_from_df_segmented(
    df_segmented: pd.DataFrame, phonetics: list[list[str]], target_word_idx: int
) -> pd.DataFrame:
    """Extract a word from a df_segmented, given the phonetics and the target word index

    Arguments
    ---------
    df_segmented : pd.DataFrame
        The output of prediction and forced alignment
    phonetics : list[list[str]]
        The phonetic transcription of the audio, as a list of lists of phones
    target_word_idx : int
        The index of the target word in the phonetic transcription
    """
    start_idx = sum([len(p) for p in phonetics][:target_word_idx])
    end_idx = sum([len(p) for p in phonetics][: target_word_idx + 1])
    df_word = df_segmented[start_idx:end_idx].copy()
    return df_word


class Wav2Vec2ForFramePrediction:
    """
    A class for predicting phonemes from audio using the Wav2Vec2 model for frame prediction.

    Attributes:
    - alphabet (list): A list of phoneme labels. Typically, the CMU phone set, or the MFA IPA phone set (derived from a dataset, so its completeness can depend on the languages used)
    - collapse_method (str): The method used to collapse repeated phoneme probability vectors. (mean, i.e. averaging them is used)
    - w2v2_model_path (str): The file path of the Wav2Vec2 model.
    - w2v2_model_format (str): The format of the Wav2Vec2 model ("torch" or "onnx"). ONNX is a tool for compressing models, and maing them faster. I used the quantization on the basis model.
    - reducer (PCA): A PCA dimensionality reduction model. I tried also UMAP, but slower and I couldn't achieve better results
    - frame_classifier (KNeighborsClassifier): A K-nearest neighbors classifier for frame classification. It could be any other sklearn classifier, I chose this one, but others could lead to better performance.
    - status (str): The status of the model. "success", or "success: ..." when a special case was found, or "error: ..." when an error in the code occurred.

    Methods:
    - __init__(self, alphabet, collapse_method, w2v2_model_path, w2v2_model_format, reducer, frame_classifier):
        Initializes the Wav2Vec2ForFramePrediction object with the specified attributes.
    - save(self, out_path, name): Saves the model to disk.
    - load(self, out_path, name): Loads a saved model from disk.
    - get_last_hidden_state(self, s, fs): Returns the last hidden state output of the Wav2Vec2 model given an audio sample.
    - reduce_lhs_dimension(self, lhs): Reduces the dimensionality of the last hidden state output using PCA.
    - fit(self, X, y): Trains the dimensionality reduction model and frame classifier using the given training data.
    - predict_phone_prob_matrix(self, s, fs): Computes the probability matrix of each frame corresponding to every phoneme for an audio sample.
    - audio_to_phone_prob_matrix(self, audio, phonetics, max_speech_rate, mode): Converts an audio sample to a probability matrix of phonemes.
    - audio_to_phone_prob_df(self, audio, phonetics, max_speech_rate, mode): Converts an audio sample to a DataFrame of phoneme probabilities.
    - max_posterior_phone_df(self, phone_prob_df, proba_thresh): Computes the maximum posterior probability of each phoneme from a DataFrame of phoneme probabilities.
    - phone_prob_matrix_segmentation(self, phone_prob_matrix, phoneme_list): Performs forced alignment segmentation on a probability matrix of phonemes.
    """

    SILENCE = "[SIL]"

    def __init__(
        self,
        alphabet: Sequence[str],
        collapse_method: str = "mean",
        w2v2_model_path: str = "hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        w2v2_model_format: str = "torch",
        reducer: TransformerMixin | None = None,
        frame_classifier: ClassifierMixin | None = None,
    ):
        """
        Initializes the Wav2Vec2ForFramePrediction object with the specified attributes.

        Args:
        - alphabet (list): A list of phoneme labels.
        - collapse_method (str): The method used to collapse repeated phoneme labels.
        - w2v2_model_path (str): The file path of the Wav2Vec2 model.
        - w2v2_model_format (str): The format of the Wav2Vec2 model ("torch" or "onnx").
        - reducer (PCA): A PCA dimensionality reduction model, could be another sklearn reduction model.
        - frame_classifier (KNeighborsClassifier): By default, a K-nearest neighbors classifier for frame classification. It sould be any other sklearn mclassifier.
        """
        self.fs: float = 16000
        self.time_per_output: float = 0.02

        self.alphabet = alphabet
        self.collapse_method = collapse_method
        self.forced_aligner = dtw_forced_aligner(
            alphabet, collapse_method=collapse_method
        )

        self.alphabet_with_silence = self.add_silence(alphabet)
        self.id_to_p = dict(enumerate(self.alphabet_with_silence))
        self.p_to_id = {p: i for i, p in enumerate(self.alphabet_with_silence)}

        self.reducer = reducer or PCA(n_components=0.95, random_state=42)
        self.frame_classifier = frame_classifier or KNeighborsClassifier(10)

        # import Wav2Vec2 feature extractor
        self.w2v2_model_format = w2v2_model_format
        if w2v2_model_format == "torch":
            self.model = Wav2Vec2Model.from_pretrained(
                w2v2_model_path, output_hidden_states=True
            )
        else:
            import onnxruntime as rt

            quantized_model_name = "last_hidden_state.quant.onnx"
            quantized_model_path = '/'.join(["hf_models", quantized_model_name])
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = (
                rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.session = rt.InferenceSession(
                quantized_model_path, sess_options, providers=['CPUExecutionProvider']
            )

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    @classmethod
    def add_silence(cls, alphabet: Sequence[str]) -> Sequence[str]:
        """
        Adds the silence token to the alphabet.

        Args:
        - alphabet (list): A list of phone labels.

        Returns:
        - The alphabet with the silence token.
        """
        return [*alphabet, cls.SILENCE]

    def save(self, out_path='models', name='model_mailabs_pca_0.95_knn_10_w'):
        """
        Saves the model to disk.

        Args:
        - out_path (str): The output directory path.
        - name (str): The name of the saved model.
        """
        path = os.path.join(out_path, name)
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(self.reducer, open(path + "/reducer.p", "wb"))
        pickle.dump(self.frame_classifier, open(path + "/frame_classifier.p", "wb"))

    def load(self, out_path='models', name='model_mailabs_pca_0.95_knn_10_w'):
        """
        Loads a saved model from disk.

        Args:
        - out_path (str): The output directory path.
        - name (str): The name of the saved model.
        """
        path = os.path.join(out_path, name)
        self.reducer = pd.read_pickle(path + "/reducer.p")
        self.frame_classifier = pd.read_pickle(path + "/frame_classifier.p")

    # get output from an audio sample in w2v2 feature extractor
    def get_last_hidden_state(self, s, fs):
        """
        Returns the last hidden state output of the Wav2Vec2 model given an audio sample.

        Args:
        - s: The audio sample.
        - fs: The sampling rate of the audio.

        Returns:
        - The last hidden state output of the Wav2Vec2 model.
        """
        input_values = self.processor(
            torch.tensor(s), sampling_rate=fs, return_tensors="pt"
        ).input_values.to('cpu')

        if self.w2v2_model_format == "torch":
            with torch.no_grad():
                return self.model(input_values).hidden_states[-1]  # type: ignore
        else:
            onnx_outputs = self.session.run(
                None, {self.session.get_inputs()[0].name: input_values.numpy()}
            )[0]
            return onnx_outputs

    # reduce dimension from w2v2 output
    def reduce_lhs_dimension(self, lhs):
        """
        Reduces the dimensionality of the last hidden state output using PCA.

        Args:
        - lhs: The last hidden state output of the Wav2Vec2 model.

        Returns:
        - The reduced last hidden state output.
        """
        return self.reducer.transform(lhs[0])  # type: ignore

    def fit(self, X, y):
        """
        Trains the dimensionality reduction model and frame classifier using the given training data.
        This training input data X must be the last hidden states for each audio frame. The labels are the labels (phones that must be in the alphabet used)


        Args:
        - X: The training data.
        - y: The target labels.
        """
        self.X_train = X
        self.y_train_labels = y

        self.y_train = [self.p_to_id[el] for el in y]

        logger.info('fit frame reducer...')
        self.reducer.fit(self.X_train)  # type: ignore
        logger.info('reduce training data')
        self.X_train_reduced = self.reducer.transform(self.X_train)  # type: ignore
        logger.info('fit frame classifier...')
        self.frame_classifier.fit(self.X_train_reduced, self.y_train)  # type: ignore

    # from an audio sample, computes the probability matrix of each frame corresponding to every phoneme
    def predict_phone_prob_matrix(self, s: np.ndarray, fs: float) -> np.ndarray:
        """
        Computes the probability matrix of each frame corresponding to every phoneme for an audio sample.

        Args:
        - s: The audio sample.
        - fs: The sampling rate of the audio.

        Returns:
        - The probability matrix of each frame corresponding to every phoneme.
        """
        self.timestamps = []
        start = time()
        self.lhs = self.get_last_hidden_state(s, fs)
        self.timestamps.append(time() - start)

        start = time()
        self.reduced_lhs = self.reduce_lhs_dimension(self.lhs)
        self.timestamps.append(time() - start)

        start = time()
        phone_prob_matrix = cast(
            np.ndarray, self.frame_classifier.predict_proba(self.reduced_lhs)  # type: ignore
        )
        self.timestamps.append(time() - start)

        # if during training, the classifier has not seen some of the labels, it won't be in the possible labels, and the proba matrix will have a reduced shape
        # thus here I extract indices that don't have a column in the matrix to then add rows of zeros and have a correct shape
        ids_to_add = [
            el
            for el in range(len(self.id_to_p))
            if el not in self.frame_classifier.classes_  # type: ignore
        ]

        for i in ids_to_add:
            phone_prob_matrix = np.concatenate(
                [
                    phone_prob_matrix[:, :i],
                    np.zeros((phone_prob_matrix.shape[0], 1)),
                    phone_prob_matrix[:, i:],
                ],
                axis=1,
            )

        logger.debug(
            "Times of get_last_hidden_state, reduce_lhs_dimension, classifier predict_proba:"
        )
        logger.debug(self.timestamps)
        return phone_prob_matrix

    def audio_to_phone_prob_matrix(
        self,
        audio: AudioInput,
        phonetics: str,
        min_speech_rate: float = DEFAULT_MIN_SPEECH_RATE,
        max_speech_rate: float = DEFAULT_MAX_SPEECH_RATE,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        mode: AudioMode = AudioMode.NUMPY,
    ) -> tuple[AudioLoadResult, np.ndarray | None]:
        """
        Converts an audio sample to a probability matrix of phonemes.

        Parameters
        ----------
        audio : AudioInput
            The audio sample.
        phonetics : str
            The phonetic transcription of the audio.
        min_speech_rate : float
            The minimum accepted speech rate in syllables per second.
        max_speech_rate : float
            The maximum accepted speech rate in syllables per second.
            This acts as a detector of abnormally short audio sample compared to the
            number of phonemes to detect very short audios with nothing useful in them.
        silence_threshold : float
            The threshold of silence in dBFS.
        mode : AudioMode
            The mode of the audio data.

        Returns
        -------
        audio_load : AudioLoadResult
            The status of the audio loading and check
        phone_prob_matrix : np.ndarray
            The probability matrix of phonemes.
        """
        phonetics = remove_grouping_hyphens(phonetics)
        with CodeTimer('load audio', silent=True):
            audio_load = audio_load_and_check(
                audio,
                phonetics,
                min_speech_rate=min_speech_rate,
                max_speech_rate=max_speech_rate,
                silence_threshold=silence_threshold,
                mode=mode,
                fs=self.fs,
            )
        if audio_load.status == AudioStatus.SUCCESS:
            with CodeTimer('phone_prob_matrix prediction', silent=True):
                if audio_load.waveform is None:
                    raise ValueError("phone_prob_matrix is None")
                phone_prob_matrix = self.predict_phone_prob_matrix(
                    audio_load.waveform, self.fs
                )
            return audio_load, phone_prob_matrix
        else:
            return audio_load, None

    def audio_to_phone_prob_df(
        self, audio, phonetics, max_speech_rate=8, mode: AudioMode = AudioMode.NUMPY
    ):
        """
        Converts an audio sample to a DataFrame of phoneme probabilities.

        Parameters
        ----------
        audio : type
            The audio sample.
        phonetics : str
            The phonetic transcription of the audio.
        max_speech_rate : float
            The maximum accepted speech rate in syllables per second.
        mode : AudioMode
            The mode of the audio data.

        Returns
        -------
        audio_load : AudioLoadResult
            The status of the audio loading and check.
        phone_prob_df : type
            The DataFrame of phoneme probabilities.
        """
        audio_load, phone_prob_matrix = self.audio_to_phone_prob_matrix(
            audio, phonetics, max_speech_rate=max_speech_rate, mode=mode
        )
        if audio_load.status == AudioStatus.SUCCESS and phone_prob_matrix is not None:
            phone_prob_df = pd.DataFrame(phone_prob_matrix)
            phone_prob_df.columns = self.alphabet_with_silence  # type: ignore
            return audio_load, phone_prob_df
        else:
            return audio_load, None

    def max_posterior_phone_df(
        self, phone_prob_matrix: np.ndarray, proba_thresh: float = 0.5
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes the maximum posterior probability of each phoneme from a DataFrame
        of phoneme probabilities.

        Args:
        - phone_prob_matrix (np.ndarray): The probability matrix of phones.
        - proba_thresh (float): The probability threshold, by default 0.5.

        Returns:
        - max_posterior_df: The DataFrame of maximum posterior probabilities.
        - max_posterior_df_filtered_processed: The processed (no silences, collapsed
            per phoneme instead of per frame) DataFrame of maximum posterior probabilities.
        - max_posterior_df_filtered_processed_threshed: The thresholded
          (remove phones with to low posterior probability) DataFrame of maximum posterior probabilities.
        """
        max_idxs = np.argmax(phone_prob_matrix, axis=1)

        max_posterior_df = pd.DataFrame({"max_idx": max_idxs})
        max_posterior_df["phone"] = max_posterior_df.max_idx.apply(
            lambda x: self.alphabet_with_silence[x]
        )
        max_posterior_df["proba"] = phone_prob_matrix.max(axis=1)

        max_posterior_df_filtered = max_posterior_df[
            max_posterior_df.phone != self.SILENCE
        ].copy()

        # group consecutive duplicates
        max_posterior_df_filtered["idx_for_merging"] = (
            max_posterior_df_filtered["max_idx"].diff().ne(0).cumsum()
        )

        def merging(x):
            d = {}
            d["phone"] = x["phone"].iloc[0]
            d["max_idx"] = x["max_idx"].iloc[0]
            d["n_frames"] = len(x)
            d["probas"] = x["proba"].tolist()
            d["max_proba"] = x["proba"].max()
            d["mean_proba"] = np.mean(d["probas"])
            return pd.Series(d)

        max_posterior_df_filtered_processed = (
            max_posterior_df_filtered.groupby("idx_for_merging")
            .apply(merging)
            .reset_index(drop=True)
        )
        max_posterior_df_filtered.drop(columns="idx_for_merging", inplace=True)

        max_posterior_df_filtered_processed_threshed = (
            max_posterior_df_filtered_processed[
                max_posterior_df_filtered_processed.max_proba > proba_thresh
            ].copy()
        )

        # regroup once more, since after thresholding we may once again have duplicates
        max_posterior_df_filtered_processed_threshed["idx_for_merging"] = (
            max_posterior_df_filtered_processed_threshed["max_idx"].diff().ne(0).cumsum()
        )

        def merging2(x):
            d = {}
            d["phone"] = x["phone"].iloc[0]
            d["n_frames"] = x["n_frames"].sum()
            d["probas"] = sum(x["probas"], [])
            d["max_proba"] = x["max_proba"].max()
            # do a weighted average by number of frames from the previous grouping
            d["mean_proba"] = np.mean(x["n_frames"] * x["mean_proba"]) / d["n_frames"]
            return pd.Series(d)

        max_posterior_df_filtered_processed_threshed = (
            max_posterior_df_filtered_processed_threshed.groupby("idx_for_merging")
            .apply(merging2)
            .reset_index(drop=True)
        )

        return (
            max_posterior_df,
            max_posterior_df_filtered_processed,
            max_posterior_df_filtered_processed_threshed,
        )

    def phone_prob_matrix_segmentation(
        self, phone_prob_matrix: np.ndarray, phone_list: list[str]
    ) -> tuple[pd.DataFrame, float | None]:
        """
        Performs forced alignment segmentation on a probability matrix of phones.

        Given a phone probability matrix of shape T x (N+1), where T is the number
        of frames and N the number of target phones, and a list of M target phones,
        return a dataframe with M rows, where the phone matrix has been segmented
        according to the DTW path.

        Args:
        - phone_prob_matrix: The predicted probability matrix of phones.
        - phone_list: The target list of phones.

        Returns:
        - df_segmented: The segmented DataFrame, with columns:
            - phones: the expected phones that the alignment was done to
            - start_idx: the starting frame index of the phone
            - end_idx: the ending frame index of the phone
            - pred_phones_audio: predicted phone for each aligned group
                (= the phone with the max. probability after pooling)
            - proba_means: the phone probability vectors pooled across
                the frames of the aligned group
            - GT_proba: the ground truth phone probability from the pooled vector
            - pred_proba: the probability value of the predicted phone
            - start: the start time of the phone in seconds
            - end: the end time of the phone in seconds
            - n_times: the number of duplicates collapsed
        - dtw_cost: The final DTW alignment cost value (or None if alignment failed)
        """
        with CodeTimer('DTW', silent=True):
            df_segmented, dtw_cost = self.forced_aligner.probas_to_df_segmented(
                phone_prob_matrix, phone_list, time_per_output=self.time_per_output
            )
        return df_segmented, dtw_cost


def train_Wav2Vec2ForFramePrediction_model():
    from sklearn.discriminant_analysis import (
        QuadraticDiscriminantAnalysis,
        LinearDiscriminantAnalysis,
    )
    from sklearn.linear_model import LogisticRegression
    from src.load_data import df_all_frames_to_X_y

    df_all_instances_select_with_silences = pd.read_pickle(
        'df_all_instances_select_with_silences.pkl'
    )
    X, y = df_all_frames_to_X_y(df_all_instances_select_with_silences)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance')
    )
    model.fit(X, y)
    model.save(name='model_mailabs_equilibrated_pca_95_knn_10_w')

    df_all_instances_select_with_silences = pd.read_pickle(
        'df_all_instances_select_with_silences_no_CH_JH.pkl'
    )
    X, y = df_all_frames_to_X_y(df_all_instances_select_with_silences)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance')
    )
    model.fit(X, y)
    model.save(name='model_mailabs_equilibrated_pca_95_knn_10_w_no_CH_JH')

    # Basis model Wav2Vec2ForFramePrediction, a 10-NN classifier weighted with distances
    df_all_frames = pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance')
    )
    model.fit(X, y)
    model.save(name='model_mailabs_pca_0.95_knn_10_w')

    df_all_instances_select_with_silences = pd.read_pickle(
        'df_all_instances_select_with_silences.pkl'
    )
    X, y = df_all_frames_to_X_y(df_all_instances_select_with_silences)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet,
        reducer=PCA(n_components=0.95, random_state=42),
        frame_classifier=LinearDiscriminantAnalysis(),
    )
    model.fit(X, y)
    model.save(name='model_mailabs_equilibrated_pca_99_lda')

    # Basis model Wav2Vec2ForFramePrediction, a 10-NN classifier weighted with distances
    df_all_frames = pd.read_pickle('df_all_frames_MAILABS_train_ipa.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        ipa_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance')
    )
    model.fit(X, y)
    model.save(name='model_mailabs_pca_95_knn_10_w_ipa')

    # Basis model Wav2Vec2ForFramePrediction, but a 10-NN classifier weighted with distances
    df_all_frames = pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet,
        reducer=PCA(n_components=0.99, random_state=42),
        frame_classifier=KNeighborsClassifier(5, weights='distance', metric='cosine'),
    )
    model.fit(X, y)
    model.save(name='model_mailabs_pca_99_knn_5_cos_w')

    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    df_all_frames = pd.read_pickle('df_all_frames_commonvoice_en_dev.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet,
        reducer=PCA(n_components=0.99, random_state=42),
        frame_classifier=LinearDiscriminantAnalysis(),
    )
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_lda')

    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    df_all_frames = pd.read_pickle('df_all_frames_commonvoice_en_dev.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet,
        reducer=PCA(n_components=0.99, random_state=42),
        frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'),
    )
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_knn')

    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    df_all_frames = pd.read_pickle('df_all_frames_commonvoice_en_dev_uk_us_ca_n_1000.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet,
        reducer=PCA(n_components=0.99, random_state=42),
        frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'),
    )
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_knn_uk_us_ca_n_1000')

    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    df_all_frames = pd.read_pickle('df_all_frames_commonvoice_en_dev_uk_us_ca_n_1000.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet,
        reducer=PCA(n_components=0.99, random_state=42),
        frame_classifier=LinearDiscriminantAnalysis(),
    )
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_lda_uk_us_ca_n_1000')

    # model Wav2Vec2ForFramePrediction, but a 10-NN classifier weighted with distances, UK US FR ES, IPA
    df_all_frames = pd.read_pickle('df_all_frames_MAILABS_UK_US_FR_ES_train_ipa.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        ipa_alphabet,
        frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'),
    )
    model.fit(X, y)
    model.save(name='model_mailabs_pca_0.95_knn_10_cos_w_UK_US_FR_ES')

    # Basis model Wav2Vec2ForFramePrediction, PCA 99% variance, logistic regression
    df_all_frames = pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet,
        reducer=PCA(n_components=0.99, random_state=42),
        frame_classifier=LogisticRegression(max_iter=1000),
    )
    model.fit(X, y)
    model.save(name='model_mailabs_pca_99_logistic_regression')

    # model Wav2Vec2ForFramePrediction, PCA 99% variance, logistic regression, UK US FR ES, IPA
    df_all_frames = pd.read_pickle('df_all_frames_MAILABS_UK_US_FR_ES_train_ipa.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        ipa_alphabet,
        reducer=PCA(n_components=0.99, random_state=42),
        frame_classifier=LogisticRegression(max_iter=1000),
    )
    model.fit(X, y)
    model.save(name='model_mailabs_pca_99_logistic_regression_UK_US_FR_ES')

    # estimators=[('5-NN cosine metric weighted', KNeighborsClassifier(5, weights='distance', metric='cosine')), ('LogisticRegression', LogisticRegression(max_iter=1000))]
    # eclf = VotingClassifier(estimators=estimators, voting='soft', weights=[1,5])
    # reducer=PCA(n_components=0.99, random_state=42)


def inference_demo():
    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction
    from src.load_data import load_libri_dataset, load_libri_dataset_audio_timings

    # --------------- Inference demo --------------------

    # phoneme predictions on a train dataset with forced alignment
    # comment for cmu or ipa
    df_t_train, df_t_test = load_libri_dataset()
    data = load_libri_dataset_audio_timings(df_t_test)

    # default_model_cmu = Wav2Vec2ForFramePrediction(cmu_alphabet)
    model = Wav2Vec2ForFramePrediction(cmu_alphabet, w2v2_model_format="onnx")
    # model.load(name='model_mailabs_pca_0.95_knn_10_w')
    model.load(name='model_mailabs_equilibrated_pca_95_knn_10_w')

    from src.text_processing import remove_stress_annots

    # phoneme predictions on a single audio sample with forced alignment
    phone_prob_matrix = model.predict_phone_prob_matrix(data.s.iloc[0], 16000)
    df_segmented, _ = model.phone_prob_matrix_segmentation(
        phone_prob_matrix, data.cmu_phones.iloc[0]
    )

    model = Wav2Vec2ForFramePrediction(
        cmu_stressed_alphabet,
        reducer=PCA(n_components=0.95, random_state=42),
        frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'),
        w2v2_model_format="onnx",
    )
    model.load(name='model_mailabs_equilibrated_stressed_pca_95_knn_10_cos_w')
    phone_prob_matrix = model.predict_phone_prob_matrix(data.s.iloc[0], 16000)
    df_segmented, _ = model.phone_prob_matrix_segmentation(
        phone_prob_matrix, data.phone_df.iloc[0].phone.tolist()
    )

    # from src.label_data_processing import synth_words_data
    from src.audio_processing import read_audio_file
    from src.text_processing import prefill_for_sentence

    import matplotlib.pyplot as plt
    import seaborn as sns
    import re

    # df=synth_words_data()

    word = "sister"
    phonetics = prefill_for_sentence(word)['phonetics']
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_" + word + ".mp3"
    s, fs = read_audio_file(path, fs=16000)

    phone_list = re.sub("[0-9]", "", phonetics).replace('|', "_").split('_')

    # phoneme predictions on a single audio sample with forced alignment
    prob_matrix = model.predict_phone_prob_matrix(s, 16000)

    latentogram = model.reducer.transform(model.lhs[0])  # type: ignore
    df_segmented.start_idx.tolist()

    # to have horizontal line in white in the heatmap at the phone starts, I put a 6
    latentogram[df_segmented.start_idx.tolist(), :] = 10

    plt.clf()
    sns.heatmap(latentogram)
    plt.savefig('w2v_latentogram_reduced.png')

    return df_segmented, prob_matrix


def use_tests():
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.neural_network import MLPClassifier
    # from sklearn.svm import SVC
    # from sklearn.gaussian_process import GaussianProcessClassifier
    # from sklearn.gaussian_process.kernels import RBF
    # from sklearn.naive_bayes import GaussianNB

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    from sklearn.linear_model import LogisticRegression

    # backup different possible reducers
    # if reducer == "umap":
    #     # parameters advised for clustering: https://umap-learn.readthedocs.io/en/latest/clustering.html
    #     self.reducer = UMAP(n_components=target_dim, n_neighbors=30, min_dist=0.0, random_state=42)
    # elif reducer == "parametric_umap":
    #     from umap.parametric_umap import ParametricUMAP
    #     self.reducer = ParametricUMAP(n_components=target_dim, n_neighbors=30, min_dist=0.0, random_state=42)
    # elif reducer == "pca":
    #     self.reducer = PCA(n_components=target_dim, random_state=42)

    from src.load_data import (
        load_dataset_MAILABS,
        load_dataset_commonvoice,
        build_df_all_frames,
        df_all_frames_to_X_y,
    )
    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction

    from scripts.frame_classifiers_experiments import (
        build_frame_dataset,
        build_frame_test_set,
    )

    df_all_frames_select_stressed_no_CH_JH = build_frame_dataset()
    df_all_frames_no_CH_JH = build_frame_test_set()

    X, y = df_all_frames_to_X_y(df_all_frames_select_stressed_no_CH_JH)
    model = Wav2Vec2ForFramePrediction(
        cmu_stressed_alphabet,
        reducer=PCA(n_components=0.95, random_state=42),
        frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'),
    )
    model.fit(X, y)
    model.save(name='model_mailabs_equilibrated_stressed_pca_95_knn_10_cos_w')

    df_t = load_dataset_commonvoice(
        lang_codes=['en'],
        path='./data/cv-corpus-10.0-delta-2022-07-04',
        split="dev",
        phone_set='CMU',
    )
    df_t.accents.unique()
    accents = ['United States English']
    # accents=['England English']
    df_t = df_t[df_t.accents.isin(accents)]
    df_all_frames = build_df_all_frames(df_t.sample(frac=1, random_state=0), 'phone')
    df_all_frames.to_pickle('df_all_frames_commonvoice_en_dev_us.pkl')

    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    # df_all_frames=pd.read_pickle('df_all_frames_commonvoice_en_dev_uk.pkl')

    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet,
        reducer=PCA(n_components=0.99, random_state=42),
        frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'),
    )
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_knn_us')

    # ----------------------------

    # load UK US FR ES in MFA_IPA
    df_t_train_all = load_dataset_MAILABS(
        ['en_US', 'en_UK', 'es_ES', 'fr_FR'],
        path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS',
        phone_set='MFA_IPA',
    )
    df_t_train_all = df_t_train_all.dropna()
    df_all_frames_all = build_df_all_frames(df_t_train_all, 'phone')
    df_all_frames_all.to_pickle('df_all_frames_MAILABS_UK_US_FR_ES_train_ipa.pkl')

    from src.wav2vec2_utils import plot_reduction

    df_all_frames_all['average_vector'] = df_all_frames_all['vector']
    df_all_frames_all.sample(frac=1, random_state=1)
    plot_reduction(
        df_all_frames_all.sample(frac=1, random_state=1),
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_ft_space_phoneme',
        legend_label='phoneme',
    )
    plot_reduction(
        df_all_frames_all.sample(frac=1, random_state=1),
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_ft_space_genre',
        legend_label='genre',
    )
    plot_reduction(
        df_all_frames_all.sample(frac=1, random_state=1),
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_ft_space_speaker',
        legend_label='speaker',
    )
    plot_reduction(
        df_all_frames_all.sample(frac=1, random_state=1),
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_ft_space_language_code',
        legend_label='language_code',
    )

    df_all_frames_select_stressed_no_CH_JH[
        'average_vector'
    ] = df_all_frames_select_stressed_no_CH_JH['vector']
    plot_reduction(
        df_all_frames_select_stressed_no_CH_JH.sample(frac=1, random_state=1),
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_ft_space_phoneme_selected_CMU_stressed',
        legend_label='phoneme',
    )

    # load UK US in MFA_IPA
    # df_t_train = load_dataset_MAILABS(['en_US', 'en_UK'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', phone_set='MFA_IPA')
    # df_t_train=df_t_train.dropna()
    # df_all_frames = build_df_all_frames(df_t_train, 'phone')
    # df_all_frames.to_pickle('df_all_frames_MAILABS_train_ipa.pkl')

    # load US english in CMU
    df_t_train = load_dataset_MAILABS(
        ['en_US'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS'
    )
    df_t_train = df_t_train.dropna()

    df_all_frames = build_df_all_frames(df_t_train, 'phone')
    df_all_frames.to_pickle('df_all_frames_MAILABS_US_train.pkl')

    # load UK US english in CMU
    df_t_train = load_dataset_MAILABS(
        ['en_US', 'en_UK'],
        path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS',
    )
    df_t_train = df_t_train.dropna()

    df_all_frames = build_df_all_frames(df_t_train, 'phone')
    df_all_frames.to_pickle('df_all_frames_MAILABS_train.pkl')

    df_all_frames = build_df_all_frames(
        df_t_train, 'phone', model_path="hf_models/facebook/wav2vec2-base-960h"
    )
    df_all_frames.to_pickle('df_all_frames_MAILABS_train_w2v_base.pkl')

    df_all_frames = build_df_all_frames(
        df_t_train, 'phone', model_path="hf_models/facebook/wav2vec2-large-xlsr-53"
    )
    df_all_frames.to_pickle('df_all_frames_MAILABS_train_w2v_xlsr_no_ft.pkl')

    df_t_train_all = load_dataset_MAILABS(
        ['en_US', 'en_UK', 'es_ES', 'fr_FR'],
        path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS',
        phone_set='MFA_IPA',
    )
    df_all_frames = build_df_all_frames(
        df_t_train_all, 'phone', model_path="hf_models/facebook/wav2vec2-large-xlsr-53"
    )
    df_all_frames.to_pickle(
        'df_all_frames_MAILABS_UK_US_FR_ES_train_ipa_w2v_xlsr_no_ft.pkl'
    )

    from src.wav2vec2_utils import plot_reduction

    df_all_frames['average_vector'] = df_all_frames['vector']
    plot_reduction(
        df_all_frames,
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_no_ft_space_phoneme',
        legend_label='phoneme',
    )
    plot_reduction(
        df_all_frames,
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_no_ft_space_genre',
        legend_label='genre',
    )
    plot_reduction(
        df_all_frames,
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_no_ft_space_speaker',
        legend_label='speaker',
    )
    plot_reduction(
        df_all_frames,
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_no_ft_space_language_code',
        legend_label='language_code',
    )

    records = []
    for fn in df_all_frames.filename.unique():
        df_all_frames_fn = df_all_frames[df_all_frames.filename == fn]

        keys = ['language_code', 'genre', 'speaker', 'filename']
        d = {}
        for k in keys:
            d[k] = df_all_frames_fn.iloc[0][k]

        d['average_vector'] = df_all_frames_fn.vector.mean()
        records.append(d)
    df_all_sentences = pd.DataFrame.from_records(records)

    # plot_reduction(df_all_sentences, reduction_technique='umap', base_name='plots/w2v_xlsr_no_ft_sentence_space_phoneme', legend_label='phoneme')
    plot_reduction(
        df_all_sentences,
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_no_ft_sentence_space_genre',
        legend_label='genre',
    )
    plot_reduction(
        df_all_sentences,
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_no_ft_sentence_space_speaker',
        legend_label='speaker',
    )
    plot_reduction(
        df_all_sentences,
        reduction_technique='umap',
        base_name='plots/w2v_xlsr_no_ft_sentence_space_language_code',
        legend_label='language_code',
    )

    # # Basis model Wav2Vec2ForFramePrediction, in CMU phoneme set, PCA reduction at 95% variance, and a 10-NN classifier
    # df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    # X, y = df_all_frames_to_X_y(df_all_frames)
    # model = Wav2Vec2ForFramePrediction(0.95,'cmu', reducer="pca")
    # model.fit(X, y)

    # Basis model Wav2Vec2ForFramePrediction, 10-NN classifier weighted with distances, US frames
    df_all_frames = pd.read_pickle('df_all_frames_MAILABS_US_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance')
    )
    model.fit(X, y)
    model.save(name='model_mailabs_pca_0.95_knn_10_w_US')

    # Basis model Wav2Vec2ForFramePrediction, but a 10-NN classifier weighted with distances
    df_all_frames = pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance')
    )
    model.fit(X, y)
    model.save(name='model_mailabs_pca_0.95_knn_10_w')

    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance')
    )
    model.load(name='model_mailabs_pca_0.95_knn_10_w')

    # Basis model Wav2Vec2ForFramePrediction with 10-NN classifier weighted with distances, with w2v2 base model instead of wlsr espeak ft
    df_all_frames = pd.read_pickle('df_all_frames_MAILABS_train_w2v_base.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet,
        w2v2_model_path="hf_models/facebook/wav2vec2-base-960h",
        frame_classifier=KNeighborsClassifier(10, weights='distance'),
    )
    model.fit(X, y)
    model.save(name='model_w2v_base_mailabs_pca_0.95_knn_10_w')

    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance')
    )
    model.load(name='model_w2v_base_mailabs_pca_0.95_knn_10_w')

    df_all_instances = pd.read_pickle('df_all_phonemes_instances_MAILABS_train.pkl')
    df_all_instances['sum'] = df_all_instances.iloc[:, 1].apply(lambda r: r.sum())
    df_all_instances = df_all_instances.dropna()
    import numpy as np

    X_train = np.array(list(df_all_instances.iloc[:, 1].values))
    # id_to_p={i:p for i,p in enumerate(cmu_alphabet+['[SIL]'])}
    # p_to_id={p:i for i,p in enumerate(cmu_alphabet+['[SIL]'])}
    # y_train=df_all_instances.phoneme.apply(lambda r: p_to_id[r]).values
    y_train = list(df_all_instances.phoneme.values)

    default_model_ipa = Wav2Vec2ForFramePrediction(ipa_alphabet)
    default_model_ipa.load(name='model_mailabs_pca_95_knn_10_w_ipa')

    # https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
    from sklearn.ensemble import VotingClassifier

    #  with ensemble
    estimators = [
        (
            '10 Nearest Neighbors',
            KNeighborsClassifier(n_neighbors=10, weights='distance'),
        ),
        ('QDA', QuadraticDiscriminantAnalysis()),
    ]
    eclf = VotingClassifier(
        estimators=estimators, voting='soft', weights=[1 for _ in estimators]
    )
    model = Wav2Vec2ForFramePrediction(cmu_alphabet, frame_classifier=eclf)
    model.fit(X, y)

    # pickle.dump(model,open('model_mailabs_pca_0.95_eclf_knn_10_linear_svm_qda.pkl','wb'))

    ###

    model = Wav2Vec2ForFramePrediction(
        cmu_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance')
    )
    model.fit(X, y)
    # model.fit_phoneme(X_train, y_train)
    # pickle.dump(model,open('model_mailabs_pca_0.95_eclf_knn_10_weighted_dist.pkl','wb'))
