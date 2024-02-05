#!/usr/bin/env python

import argparse
import ast
import logging
from pathlib import Path

import librosa
import pandas as pd

from flowspeech.DL_accuracy_performance import (
    final_ed_confusions_for_actor_recordings,
    final_ed_confusions_from_audiobook_data,
    final_ed_s_confusions_on_synth_words,
    final_s_confusions_on_synth_words,
    h_sound_confusions_on_synth_words,
    phoneme_confusions,
    plot_confusion_results,
    pronunciation_aspects_from_audiobook_data,
    start_end_consonant_clusters_on_synth_words,
    vowels_confusions_actor_recordings,
    vowels_confusions_user_recordings,
    vowels_consonants_confusions_from_audiobook_data,
)
from flowspeech.DL_speech_tech import StressCategory, default_model
from flowspeech.label_data_processing import actor_recordings
from flowspeech.performance_functions import (
    final_s_from_audiobook_data,
    pContrast_for_actor_recordings,
    pContrast_from_audiobook_data,
    pContrast_on_synth_words,
    stress_GE_performance_test,
)
from flowspeech.pronunciation_dictionaries import cmu_consonants, cmu_vowels

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Running the performance pipeline from Noé (as best as we can make it work)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./plots", help="Output folder for plots"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    model = default_model

    #### Vowels confusions ####

    logger.info("Vowels confusions on synth words")
    phoneme_confusions(
        model=model,
        phonemes=cmu_vowels,
        performance_function=pContrast_on_synth_words,
        n=10,
        name=str(output_folder / "vowels_confusions_on_synth_words"),
    )

    # NOTE: could also use vowels_confusions_actor_recordings + plot_confusion_results
    logger.info("Vowels confusions for actor recordings")
    phoneme_confusions(
        model=model,
        phonemes=cmu_vowels,
        performance_function=pContrast_for_actor_recordings,
        name=str(output_folder / "vowels_confusions_for_actor_recordings"),
    )

    # NOTE: could also use vowels_consonants_confusions_from_audiobook_data + plot_confusion_results
    logger.info("Vowels confusions from audiobook data")
    data_set = "test-clean"
    n = 100
    # data_set = "dev-clean"
    # n = 10
    phoneme_confusions(
        model=model,
        phonemes=cmu_vowels,
        performance_function=pContrast_from_audiobook_data,
        n=n,
        data_set=data_set,
        name=str(output_folder / "vowels_confusions_from_audiobook_data"),
    )

    #### Consonants confusions ####

    logger.info("Consonants confusions on synth words")
    phoneme_confusions(
        model=model,
        phonemes=cmu_consonants,
        performance_function=pContrast_on_synth_words,
        n=10,
        name=str(output_folder / "consonants_confusions_on_synth_words"),
    )

    logger.info("Consonants confusions for actor recordings")
    phoneme_confusions(
        model=model,
        phonemes=cmu_consonants,
        performance_function=pContrast_for_actor_recordings,
        name=str(output_folder / "consonants_confusions_for_actor_recordings"),
    )

    # NOTE: could also use vowels_consonants_confusions_from_audiobook_data + plot_confusion_results
    logger.info("Consonants confusions from audiobook data")
    data_set = "test-other"
    n = 10
    phoneme_confusions(
        model=model,
        phonemes=cmu_consonants,
        performance_function=pContrast_from_audiobook_data,
        n=n,
        data_set=data_set,
        name=str(output_folder / "consonants_confusions_from_audiobook_data"),
    )

    #### Final -ed confusions ####

    logger.info("Final -ed and -s confusions on synth words")
    final_ed_s_confusions_on_synth_words(
        model=model, n=10, output_folder=str(output_folder)
    )

    data_set = "dev-clean"
    n = 10
    logger.info("Final -ed confusions from audiobook data")
    _, results = final_ed_confusions_from_audiobook_data(
        n=n, model=model, data_set=data_set
    )
    plot_confusion_results(
        results,
        name=str(
            output_folder
            / (
                "termination_contrast_audiobook_no_D_T_dis_"
                + data_set
                + "_w2v_n_"
                + str(n)
            )
        ),
    )

    logger.info("Final -ed confusions for actor recordings")
    _, results = final_ed_confusions_for_actor_recordings(model=model)
    plot_confusion_results(
        results,
        name=str(
            output_folder
            / "termination_confusions_for_actor_recordings_D_T_dis_w2v_AH_D_post_corr"
        ),
    )

    #### Final -s ####

    # FIXME: not working
    # logger.info("Final -s confusions from audiobook data")
    # data_set = "dev-clean"
    # n = 10
    # _, results = final_s_from_audiobook_data(data_set=data_set, n=n, model=model)
    # plot_confusion_results(
    #     results,
    #     name=str(
    #         output_folder
    #         / ("termination_contrast_audiobook_S" + data_set + "_w2v_n_" + str(n))
    #     ),
    # )

    #### Initial h- ####

    # initial h-, final -s
    # FIXME: not working
    # pronunciation_aspects_from_audiobook_data

    logger.info("Initial h- confusions on synth words")
    h_sound_confusions_on_synth_words(
        n=10, model=model, name=str(output_folder / "h_sound_confusions_on_synth_words")
    )

    #### Clusters ####

    logger.info("End clusters on synth words")
    start_end_consonant_clusters_on_synth_words(
        clusters=["P_TH", "M_P_T", "N_TH_S"],
        n=10,
        model=model,
        contrast="end",
        output_folder=str(output_folder),
    )
    logger.info("Start clusters on synth words")
    start_end_consonant_clusters_on_synth_words(
        clusters=["TH_R", "P_R", "S_P_L", "S_K_R"],
        n=10,
        model=model,
        contrast="start",
        output_folder=str(output_folder),
    )

    #### Word stress ####
    # stress_GE_performance_test(level=StressCategory.WORD, model=model)

    #### Sentence stress ####
    # stress_GE_performance_test(level=StressCategory.SENTENCE)


if __name__ == "__main__":
    main()
