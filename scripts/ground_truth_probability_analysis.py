#!/usr/bin/env python

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from matplotlib import pyplot as plt
from tqdm import tqdm

from flowspeech.DL_speech_tech import default_model, validate_recording
from flowspeech.label_data_processing import actor_recordings, synth_words_data
from flowspeech.libri_phonetization_data import build_librispeech_words_df
from flowspeech.pronunciation_dictionaries import (
    cmu_consonants,
    cmu_vowels,
    remove_stress_annots,
)
from flowspeech.text_processing import (
    cmu_ensure_phonetics_consistency,
    split_phonetics_to_phones,
)
from flowspeech.wav2vec2_frame_prediction import (
    AudioMode,
    AudioStatus,
    Wav2Vec2ForFramePrediction,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# mute warnings
warnings.filterwarnings("ignore")


class DataLoader:
    """Utiliy class to load and preprocess data for performanceanalysis

    TODO: improve and refactor to performance code
    """

    CHOICES = ("actor_recordings", "audiobook_data", "synth_words")

    def __init__(self, data_type: str):
        if data_type not in self.CHOICES:
            raise ValueError(f"Invalid data type {data_type}")
        self.data_type = data_type
        match data_type:
            case "actor_recordings":
                self._getter = self._get_actors
            case "audiobook_data":
                self._getter = self._get_audiobook
            case "synth_words":
                self._getter = self._get_synth_words

    def _get_actors(self) -> pd.DataFrame:
        df = actor_recordings()
        df["phonetics"] = df.cmu_phonetics
        df["fpath"] = df.audio_file_url
        return df

    def _get_audiobook(self) -> pd.DataFrame:
        # FIXME this needs to be implemented
        return build_librispeech_words_df()

    def _get_synth_words(self) -> pd.DataFrame:
        return synth_words_data()

    def get(self, n: int | None = None) -> pd.DataFrame:
        return self._getter().sample(n, random_state=42)


def compute_predictions(
    df: pd.DataFrame, model: Wav2Vec2ForFramePrediction = default_model
) -> pd.DataFrame:
    pred_dfs = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        phonetics = cmu_ensure_phonetics_consistency(r.phonetics)
        audio_load, phone_prob_matrix = model.audio_to_phone_prob_matrix(
            r.fpath, phonetics, mode=AudioMode.FILE
        )

        if audio_load.status != AudioStatus.SUCCESS or phone_prob_matrix is None:
            logger.debug(f"Failed to process {r.audio_file_url} for {phonetics}")
            continue

        if not validate_recording(phone_prob_matrix, phonetics):
            logger.debug(f"Failed to validate {r.audio_file_url} for {phonetics}")
            continue

        phoneme_list = remove_stress_annots(split_phonetics_to_phones(phonetics))
        df_segmented, dtw_cost = model.phone_prob_matrix_segmentation(
            phone_prob_matrix, phoneme_list
        )
        if dtw_cost is None:
            logger.debug(f"Failed to segment {r.audio_file_url} for {phonetics}")

        pred_dfs.append(df_segmented)

    return pd.concat(pred_dfs)


def plot_posterior_proba_distributions(
    model: Wav2Vec2ForFramePrediction,
    results_df: pd.DataFrame,
    output_name: str,
) -> np.ndarray:
    results_df["phone_id"] = results_df.phones.apply(lambda x: model.p_to_id[x])
    n_phones = len(model.p_to_id)
    proba_matrix = np.zeros((n_phones, n_phones))
    for _, r in results_df.iterrows():
        proba_matrix[r.phone_id] += r.proba_means

    # normalize row wise to have probabilities again
    proba_matrix = proba_matrix / proba_matrix.sum(axis=1, keepdims=True)

    annot_labels = np.vectorize(lambda x: f"{x:.1f}" if x > 0.2 else "")(proba_matrix)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(proba_matrix, ax=ax, annot=annot_labels, fmt="", linewidth=0.5)

    ax.set_xticks(np.arange(n_phones) + 0.5, model.alphabet_with_silence)
    ax.set_xlabel("Detected phone")
    ax.set_yticks(np.arange(n_phones - 1) + 0.5, model.alphabet)
    ax.set_ylabel("True phone")

    fig.savefig(output_name)

    return proba_matrix


def plot_ground_truth_proba_distribution(results_df: pd.DataFrame, output_name: str):
    phone_df = results_df[["phones", "GT_proba"]]

    for label, phone_set in [("vowels", cmu_vowels), ("consonants", cmu_consonants)]:
        g = sns.displot(
            data=phone_df[phone_df.phones.isin(phone_set)],
            x="GT_proba",
            col="phones",
            col_wrap=4,
            kind="hist",
        )
        g.savefig(output_name + f"_{label}.png")


def plot_phone_detection_confusion_matrix(
    model: Wav2Vec2ForFramePrediction,
    results_df: pd.DataFrame,
    output_name: str,
) -> np.ndarray:
    confusion_matrix = sklearn.metrics.confusion_matrix(
        results_df.phones,
        results_df.pred_phones_audio,
        normalize="true",
        labels=model.alphabet_with_silence,
    )
    # remove the [SIL] token from the true labels
    confusion_matrix = confusion_matrix[:-1, :]

    n_phones = len(model.p_to_id)
    annot_labels = np.vectorize(lambda x: f"{x:.1f}" if x > 0.2 else "")(confusion_matrix)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(confusion_matrix, ax=ax, annot=annot_labels, fmt="", linewidth=0.5)

    ax.set_xticks(np.arange(n_phones) + 0.5, model.alphabet_with_silence)
    ax.set_xlabel("Detected phone")
    ax.set_yticks(np.arange(n_phones - 1) + 0.5, model.alphabet)
    ax.set_ylabel("True phone")

    fig.savefig(output_name)

    return confusion_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Study the distribution of the posterior P(true phone) over a set of recordings"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./plots", help="Output folder for plots"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        choices=DataLoader.CHOICES,
        default=DataLoader.CHOICES,
        nargs="+",
        help="Datasets to process",
    )
    parser.add_argument(
        "-n",
        "--nmax",
        type=int,
        default=None,
        help="Maximum number of recordings to process",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    model = default_model

    for data_type in args.data:
        logger.info(f"Processing {data_type}")

        data = DataLoader(data_type)
        df = data.get(n=args.nmax)

        results_df = compute_predictions(df, model=model)

        plot_ground_truth_proba_distribution(
            results_df,
            output_name=str(output_folder / f"GT_proba_distribution_{data_type}"),
        )

        plot_posterior_proba_distributions(
            model,
            results_df,
            output_name=str(output_folder / f"posterior_proba_matrix_{data_type}"),
        )

        plot_phone_detection_confusion_matrix(
            model,
            results_df,
            output_name=str(output_folder / f"confusion_matrix_{data_type}"),
        )


if __name__ == "__main__":
    main()
