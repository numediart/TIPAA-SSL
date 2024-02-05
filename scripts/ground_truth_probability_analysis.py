#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from flowspeech.DL_speech_tech import default_model, validate_recording
from flowspeech.label_data_processing import actor_recordings
from flowspeech.pronunciation_dictionaries import (
    cmu_consonants,
    cmu_vowels,
    remove_stress_annots,
)
from flowspeech.text_processing import (
    cmu_ensure_phonetics_consistency,
    split_phonetics_to_phones,
)
from flowspeech.wav2vec2_frame_prediction import AudioMode, AudioStatus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_predictions(df, model=default_model):
    pred_dfs = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        phonetics = cmu_ensure_phonetics_consistency(r.cmu_phonetics)
        audio_load, phone_prob_matrix = model.audio_to_phone_prob_matrix(
            r.audio_file_url, phonetics, mode=AudioMode.FILE
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


def plot_ground_truth_proba_distribution(results_df, output_folder="probas_actors"):
    phone_df = results_df[["phones", "GT_proba"]]

    for label, phone_set in [("vowels", cmu_vowels), ("consonants", cmu_consonants)]:
        g = sns.displot(
            data=phone_df[phone_df.phones.isin(phone_set)],
            x="GT_proba",
            col="phones",
            col_wrap=4,
            kind="hist",
        )
        g.savefig(output_folder + f"/GT_proba_distribution_{label}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Study the distribution of the posterior P(true phone) over a set of recordings"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./plots", help="Output folder for plots"
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

    df = actor_recordings()

    results_df = compute_predictions(df.iloc[: args.nmax], model=model)

    plot_ground_truth_proba_distribution(results_df, output_folder=str(output_folder))


if __name__ == "__main__":
    main()
