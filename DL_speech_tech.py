from dataclasses import dataclass
from enum import Enum, auto
import os, psutil

print_memory_usage = lambda stage: print(
    stage + ": " + str(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
)

print_memory_usage("RAM - start of DL_speech_tech")
import numpy as np
import pandas as pd
from syllabipy.sonoripy import SonoriPy
from linetimer import CodeTimer
from nltk.metrics.distance import edit_distance as levenshtein_distance

print_memory_usage("RAM - DL_speech_tech after external dependencies")
from src.audio_processing import getIntonation, getIntensity, normalize

print_memory_usage("RAM - DL_speech_tech after src.audio_processing")

from src.text_processing import (
    cmu_ensure_phonetics_consistency,
    unstress,
    split_phonetics,
    remove_stress_annots,
    drop_consecutive_duplicates,
    drop_consecutive_duplicate_elements,
    phonetics_indexed_df_from_formatted_phonetics,
)

print_memory_usage("RAM - DL_speech_tech after src.text_processing")
from src.pronunciation_dictionaries import (
    cmu_vowels,
    cmu_stressed_vowels,
    cmu_consonants,
    cmu_to_gibberish,
    cmu_diphtongs,
    cmu_alphabet,
    ipa_alphabet,
    cmu_stressed_alphabet,
)
from src.pronunciation_dictionaries import ipa_vowels, ipa_consonants, ipa_to_gibberish

print_memory_usage("RAM - DL_speech_tech after src.pronunciation_dictionaries")

from src.text_processing import remove_stress_annots

# initialize model
# from src.charsiu_utils import charsiu_phone_forced_aligner
# default_model_charsiu = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')
# print_memory_usage("RAM - DL_speech_tech after default_model_charsiu")


phoneme_GT_proba_threshold_dict = {}
default_thresh = 0.2
for k in cmu_stressed_vowels:
    phoneme_GT_proba_threshold_dict[k] = default_thresh
for k in cmu_consonants:
    phoneme_GT_proba_threshold_dict[k] = default_thresh

# phoneme_GT_proba_threshold_dict['AO0']=0.1
# phoneme_GT_proba_threshold_dict['AO1']=0.1
# phoneme_GT_proba_threshold_dict['AO2']=0.1

# model=pickle.load(open('model_mailabs_umap_2_gmm_300.pkl','rb'))

from src.wav2vec2_frame_prediction import (
    AudioInput,
    AudioLoadResult,
    AudioMode,
    AudioStatus,
    Wav2Vec2ForFramePrediction,
    extract_word,
    audio_load_and_check,
)

print_memory_usage("RAM - DL_speech_tech after wav2vec2_frame_prediction")

# default_model = Wav2Vec2ForFramePrediction(cmu_alphabet)

default_model = Wav2Vec2ForFramePrediction(cmu_alphabet, w2v2_model_format="onnx")
# default_model.load(name='model_mailabs_pca_0.95_knn_10_w')
# default_model.load(name='model_mailabs_equilibrated_pca_95_knn_10_w')
default_model.load(name='model_mailabs_equilibrated_pca_95_knn_10_w_no_CH_JH')


# default_model_ipa = Wav2Vec2ForFramePrediction(ipa_alphabet,w2v2_model_format="onnx")
# default_model_ipa.load(name='model_mailabs_pca_95_knn_10_w_ipa')
# default_model_stressed = Wav2Vec2ForFramePrediction(cmu_stressed_alphabet,w2v2_model_format="onnx")
# default_model_stressed.load(name='model_mailabs_equilibrated_stressed_pca_95_knn_10_cos_w')

# default_model_ipa = Wav2Vec2ForFramePrediction(ipa_alphabet,w2v2_model_format="onnx")
# default_model_ipa.load(name='model_mailabs_pca_95_knn_10_w_ipa')

# default_model.load(name='model_mailabs_pca_99_logistic_regression')
# default_model.load(name='model_mailabs_pca_99_knn_5_cos_w')

# default_model=default_model_charsiu

print_memory_usage("RAM - DL_speech_tech after default_model")


MAX_PER_FOR_ACCEPTANCE = 0.8


class StressCategory(Enum):
    WORD = auto()
    SENTENCE = auto()


target_accepted_alternatives = {
    'AA': ['AA', 'AO'],
    'AO': ['AA', 'AO'],
    # 'OW': ['AA', 'OW'],
    'D': ['D', 'T'],
    'Z': ['Z', 'S'],
    'S': ['Z', 'S'],
    # 'T': ['D', 'T'],
    # 'IH': ['IH', 'AH', 'EH']
}

terminations_accepted_alternatives = {
    'IH_Z': ['AH_Z', 'IH_Z', 'ER_Z', 'IY_Z'],
    'IH_D': ['AH_D', 'IH_D', 'ER_D', 'IY_D'],
}


# For final -ed and final -s, we use a termination contrast with a basis that can accept enough phonemes, I take the longest target "IH_D" or "IH_Z"
# because of this, if the target is D/T or S/Z, it is frequent to have a border effect and that the real phoneme before or after is included in the result
# I want to accept these as correct

# for "D", I want to accept anything finishing with "D" except those corresponding to "IH_D"
# but if I accept all vowels, it means I wouldn't give feedback for a mistake like "S_N_OW_EH_D" for "snowed"
# therefore, I accept only consonants and diphtongs.
# It's also more likely to have this border effect with consonant because the target is a consonant (verified experimentally looking at confusions)

# However, for the phoneme after, I want to accept both vowels and consonants

all_Z = [p + "_Z" for p in list(cmu_consonants)] + [
    "Z_" + p for p in list(cmu_consonants) + list(cmu_vowels)
]
all_S = [p + "_S" for p in list(cmu_consonants)] + [
    "S_" + p for p in list(cmu_consonants) + list(cmu_vowels)
]
terminations_accepted_alternatives["Z"] = [
    el for el in all_Z if el not in terminations_accepted_alternatives['IH_Z']
]
terminations_accepted_alternatives["S"] = [
    el for el in all_S if el not in terminations_accepted_alternatives['IH_Z']
]


all_D = [p + "_D" for p in list(cmu_consonants) + list(cmu_diphtongs)] + [
    "D_" + p for p in list(cmu_consonants) + list(cmu_vowels)
]
all_T = [p + "_T" for p in list(cmu_consonants)] + [
    "T_" + p for p in list(cmu_consonants) + list(cmu_vowels)
]
terminations_accepted_alternatives["D"] = [
    el for el in all_D if el not in terminations_accepted_alternatives['IH_D']
]
terminations_accepted_alternatives["T"] = [
    el for el in all_T if el not in terminations_accepted_alternatives['IH_D']
]

# As in target_accepted_alternatives, we want to accept ['D','T'] for 'D', we should do the same when there is a superfluous phoneme, therefore:
terminations_accepted_alternatives["D"] += terminations_accepted_alternatives["T"]


def remove_downwards_trend(y):
    if len(y) > 2:
        # Remove downwards trend
        x = range(len(y))
        linear_f = np.polyfit(x, y, 1)
        a = linear_f[0]
        b = linear_f[1]
        y = y - (a * x + b)

        # normalize between 0 and 100
        y = y - min(y)
        y = y / max(y) * 100
    else:
        y = np.array(y)
    return y.astype(int).tolist()


import math

roundup = lambda n: math.ceil(n)


def intensity_to_bin(scores, n_max=2, threshold=60):
    """Convert a list of scores to binaries according to a threshold and a maximum numbers of "1" authorized.
    All elements are 0 except the ones that verify following conditions:
    - be among the n_max highest elements
    - be above threshold

    Args:
        scores (list of ints or floats): _description_
        n_max (int, optional): _description_. Defaults to 2.
        threshold (int, optional): _description_. Defaults to 60.

    Returns:
        list of ints of floats: _description_

        Examples:
        In [79]: intensity_to_bin([30,40,41,20,10,35,20], n_max=2, threshold=60)
        Out[79]: [0, 0, 0, 0, 0, 0, 0]

        In [75]: intensity_to_bin([30,70,61,10,65,20], n_max=2, threshold=60)
        Out[75]: [0, 1, 0, 0, 1, 0]
    """
    bin_scores = np.zeros(len(scores)).astype(int).tolist()
    if len(scores) == 1:
        imax = np.argmax(scores)
        bin_scores[imax] = 1
    else:
        imaxes = (
            [np.argmax(scores)]
            if len(scores) < n_max
            else np.argpartition(scores, -n_max)[-n_max:]
        )
        for imax in imaxes:
            if scores[imax] >= threshold:
                bin_scores[imax] = 1
    return bin_scores


def predict_phone(
    forced_aligner,
    df_word,
    phonetics,
    target_word_idx,
    target_syllable_idx,
    target_phones,
    target_occurence_idx=0,
    phoneme_set=cmu_vowels,
    GT_proba_threshold=0.2,
):
    """phonetics must be formatted phonetics as a string, e.g.: 'EH1_N|D_IH0_D'"""
    # phoneme_set=[p for p in remove_stress_annots(phoneme_set)]+["[SIL]"]
    phoneme_set = [p for p in phoneme_set] + ["[SIL]"]

    # split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
    # df_word=self.predict_word(audio, split_phonetics, target_word_idx)

    if len(df_word) > 0:
        word = phonetics.split(' ')[target_word_idx]
        syllables = [syl.split('_') for syl in word.split('|')]
        syllable = syllables[target_syllable_idx]
        syl = remove_stress_annots(syllable)
        idxs_of_target_occurences = [
            i for i, p in enumerate(syl) if unstress(target_phones) == p
        ]

        if target_occurence_idx < len(idxs_of_target_occurences):
            p_idx_local = idxs_of_target_occurences[target_occurence_idx]
        else:
            # self.status="error: target_occurence_idx is out of bounds"
            phonetic_detection = float('nan')
            syl = float('nan')
            return phonetic_detection, syl

        len_previous_syllables = sum([len(el) for el in syllables[:target_syllable_idx]])
        p_idx_global = len_previous_syllables + p_idx_local

        phonetic_detection = df_word.iloc[p_idx_global].pred_phones_audio

        phoneme_set_ids = forced_aligner.labelize_phonemes(phoneme_set)
        proba_means = df_word.iloc[p_idx_global].proba_means

        # if GT_proba is beyond the threshold, we take it as prediction
        if df_word.iloc[p_idx_global].GT_proba > GT_proba_threshold:
            phonetic_detection = target_phones
        else:
            # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
            filtered_proba_means = [
                0 if i not in phoneme_set_ids else el for i, el in enumerate(proba_means)
            ]
            # if everything is 0 in the filtered proba, then we keep the phonetic detection that was in the non-filtered proba. E.g., for the word "new", imagine we target the "Y" of "N_Y_UW", but it is pronounced the british way "N_UW"
            # Then, there could be zero probability in consonants, and therefore, we keep the vowel that should be close to "UW"
            if sum(filtered_proba_means) != 0:
                phonetic_detection = forced_aligner.id_to_p[
                    np.argmax(filtered_proba_means)
                ]
        syl[p_idx_local] = phonetic_detection

    else:
        phonetic_detection = float('nan')
        syl = float('nan')
    return phonetic_detection, syl


def compute_stress_score(
    df_segmented: pd.DataFrame, audio: np.ndarray, vowels: set[str], fs: int = 16000
) -> np.ndarray:
    """Use df_segmented to have the timings of vowels and compute prosody features
    (intensity, pitch, ...) to compute a value by vowel representing a stress intensity

    Args:
        phonetics (str): formatted phonetics
        audio (np array): audio signal
        vowels: set of vowels
        fs: sampling frequency
    Returns:
        weighted_score [np array]: stress intensity score for each vowel
    """
    # select vowels
    filtered_df = df_segmented[df_segmented.phones.isin(vowels)]

    f0_samples = getIntonation(audio, fs)
    intensity = getIntensity(audio, fs)

    # extract features
    # each word start and end position expressed in samples
    start_positions_samples = (
        (round(fs * filtered_df.loc[:, 'start']) + 1).astype(int).tolist()
    )
    stop_positions_samples = round(fs * filtered_df.loc[:, 'end']).astype(int).tolist()

    # to make sure we don t go beyond the end of the signal
    assert stop_positions_samples[-1] < len(
        audio
    ), "The end of the last phoneme should be inside the signal"

    Imax, Imean, Fmax, Fmean, Dur = [], [], [], [], []
    for i in range(len(filtered_df)):
        range_vowel = range(start_positions_samples[i], stop_positions_samples[i])
        Ivowel = intensity[range_vowel]
        Fvowel = f0_samples[range_vowel]

        Imax.append(max(Ivowel))
        Imean.append(np.mean(Ivowel))
        Fmax.append(max(Fvowel))
        Fmean.append(np.mean(Fvowel))

        Dur.append(filtered_df['end'].iloc[i] - filtered_df['start'].iloc[i])

    # normalization of features (projection to [0 1] range)
    zImax = normalize(Imax)
    zImean = normalize(Imean)
    zFmax = normalize(Fmax)
    zFmean = normalize(Fmean)
    zDur = normalize(Dur)

    # combine the features
    weighted_score = (
        zImax + 0.2 * zImean + zFmax + 0.2 * zFmean + 0.8 * zDur
    ) / 3.2  # needs fine-tuning once enough user data are available - in the long term train a classifier with annotated user data

    return weighted_score


###################   Pronunciation aspect functions  ################


@dataclass
class StressAnalysisResult:
    status: str
    stress_intensities: list[int] | list[list[int]]
    stress_binaries: list[int] | list[list[int]]


def stress_from_df_segmented_audio(
    sound: np.ndarray,
    df_segmented: pd.DataFrame,
    phonetics: str,
    n_words_by_chunk: list[int] | None = None,
    level: StressCategory = StressCategory.SENTENCE,
    model: Wav2Vec2ForFramePrediction = default_model,
    vowels: set[str] = cmu_vowels,
) -> StressAnalysisResult:
    """Extract stress from a dataframe of segmented phonemes and an audio signal

    Parameters
    ----------
    sound : np.array
        audio signal
    df_segmented : pd.DataFrame
        dataframe of segmented phonemes
    phonetics : str
        Phonetics, example: "AY1 W_UH1_D L_AH1_V T_UW1 G_OW1 T_UW1 AY1|ER0|L_AH0_N_D"
    n_words_by_chunk : list[int], optional
        Number of words by chunk (used only when level="sentence")
        For the above phonetics example, would be [7]
    level: StressCategory
        level of stress extraction. Can be either "word" or "sentence"
    model : Wav2Vec2ForFramePrediction
        speech model
    vowels : set[str]
        set of vowels

    Returns
    -------
    StressAnalysisResult
        stress extraction results
        Examples:
         - for sentence stress: {"status": "success", "stress_intensities": [100], "stress_binaries": [1]}
         - for word stress: {"status": "success", "stress_intensities": [[100]], "stress_binaries": [[1]]}
    """
    if level == StressCategory.SENTENCE:
        assert (
            n_words_by_chunk is not None
        ), "n_words_by_chunk should be provided in 'sentence' mode"
        assert sum(n_words_by_chunk) == len(
            phonetics.split(' ')
        ), 'The total number of words by chunk does not correspond to the number of words in phonetics'

    vowels_df = df_segmented[df_segmented.phones.isin(vowels)]
    if len(vowels_df) == 1:
        if level == StressCategory.SENTENCE:
            return StressAnalysisResult("success", [100], [1])
        if level == StressCategory.WORD:
            return StressAnalysisResult("success", [[100]], [[1]])

    with CodeTimer('stress extraction', silent=True):
        ws = compute_stress_score(df_segmented, sound, vowels, fs=model.fs)

    # TODO: I think I should check for voiceness, but I don't know if I should do it for all vowels
    if sum([el != el for el in ws]) == len(ws):
        status = (
            "success: no voiced sound detected inside supposed vowels (no pitch detected)"
        )
        return StressAnalysisResult(status, [], [])

    phonetics_indexed_df = phonetics_indexed_df_from_formatted_phonetics(phonetics)
    is_vowel = phonetics_indexed_df.apply(lambda r: unstress(r.phones) in vowels, axis=1)
    vowels_indexed_df = phonetics_indexed_df[is_vowel].copy()

    # assert len(vowels_indexed_df) == len(ws), "n of vowels should be the same as length of vowel stresses"

    if len(vowels_indexed_df) != len(ws):
        status = "error: n of vowels should be the same as length of vowel stresses"
        print("n of vowels should be the same as length of vowel stresses")
        return StressAnalysisResult(status, [], [])

    try:
        vowels_indexed_df.loc[:, 'stress_scores'] = (100 * ws).astype(int)
    except:
        status = "error: n of vowels should be the same as length of vowel stresses"
        print("n of vowels should be the same as length of vowel stresses")
        return StressAnalysisResult(status, [], [])

    word_bins = []
    word_intensities = []
    for w_idx in vowels_indexed_df.word_idx.unique():
        word = vowels_indexed_df[vowels_indexed_df.word_idx == w_idx]
        bins = [0] * len(word)
        bins[word.stress_scores.argmax()] = 1
        word_bins.append(bins)
        word_intensities.append(word.stress_scores.tolist())

    if level == StressCategory.WORD:
        return StressAnalysisResult("success", word_intensities, word_bins)
    elif level == StressCategory.SENTENCE:
        max_word_intensities = [max(w) for w in word_intensities]
        scores_grouped_by_chunk = []
        cumsum = 0
        for n in n_words_by_chunk:
            scores_grouped_by_chunk.append(max_word_intensities[cumsum : cumsum + n])
            cumsum += n

        # I tried this on General English data, and in the end, it does not seem to improve
        # scores_grouped_by_chunk=[remove_downwards_trend(el) for el in scores_grouped_by_chunk]

        bins_by_chunk = []
        for chunk in scores_grouped_by_chunk:
            bin = intensity_to_bin(chunk, n_max=roundup(len(chunk) / 3))
            bins_by_chunk.append(bin)
        return StressAnalysisResult(
            "success", sum(scores_grouped_by_chunk, []), sum(bins_by_chunk, [])
        )
    else:
        return StressAnalysisResult(
            f"error: {level} is not a valid level in stress_from_formatted_phonetics. "
            "It has to be either 'word' or 'sentence'.",
            [],
            [],
        )


def stress_from_formatted_phonetics(
    audio: AudioInput,
    phonetics: str,
    n_words_by_chunk: list[int] | None = None,
    level: StressCategory = StressCategory.SENTENCE,
    max_speech_rate: float = 8,
    mode: AudioMode = AudioMode.FILE,
    model: Wav2Vec2ForFramePrediction = default_model,
    vowels: set[str] = cmu_vowels,
) -> StressAnalysisResult:
    phonetics = phonetics.replace('CH', 'T_SH').replace('JH', 'D_ZH')
    audio_load, phone_prob_matrix = model.audio_to_phone_prob_matrix(
        audio, phonetics, max_speech_rate=max_speech_rate, mode=mode
    )
    if audio_load.status != AudioStatus.SUCCESS:
        return StressAnalysisResult(str(audio_load.status), [], [])

    # this operation is done in audio_to_phone_prob_matrix, but I have to do it again here then for consistency
    phonetics = phonetics.replace('-', ' ').replace('{', '').replace('}', '')
    phoneme_list = phonetics.replace(' ', '_').replace('|', '_').split('_')
    df_segmented, _ = model.phone_prob_matrix_segmentation(
        phone_prob_matrix, remove_stress_annots(phoneme_list)
    )

    if model.status != AudioStatus.SUCCESS:
        return StressAnalysisResult(str(model.status), [], [])

    return stress_from_df_segmented_audio(
        audio_load.waveform,
        df_segmented,
        phonetics=phonetics,
        n_words_by_chunk=n_words_by_chunk,
        level=level,
        model=model,
        vowels=vowels,
    )


def phonemeContrast_from_df_segmented(
    df_segmented,
    phonetics='T_ER1_N_D ER0|AW1_N_D',
    target_word_idx=0,
    target_syllable_idx=0,
    target_occurence_idx=0,  # will be 0  all the time for vowels, and most of the time for consonants
    target_phones='ER1',
    alternatives=cmu_vowels,
    model=default_model,
    to_gibberish=cmu_to_gibberish,
    **kwargs,
):
    g_t = [
        to_gibberish[unstress(p)]
        for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]
    ]

    phones_by_words = [el.split('_') for el in phonetics.replace('|', '_').split(' ')]
    df_word = extract_word(df_segmented, phones_by_words, target_word_idx)
    phonetic_detection, detected_syllable = predict_phone(
        model.forced_aligner,
        df_word,
        phonetics,
        target_word_idx,
        target_syllable_idx,
        target_phones,
        target_occurence_idx,
        phoneme_set=alternatives,
    )  # , GT_proba_threshold=0.2)

    # if it's nan
    if detected_syllable != detected_syllable:
        return {
            "status": model.status,
            "phonetic_detection": "null",
            "gibberish_truth": '_'.join(g_t),
            "gibberish_detected": "null",
        }

    if 'SIL' in phonetic_detection:
        return {
            "status": model.status,
            "phonetic_detection": "null",
            "gibberish_truth": '_'.join(g_t),
            "gibberish_detected": "null",
        }

    g_d = [to_gibberish[unstress(p)] for p in detected_syllable]

    # post-correction: for the target, when we are in a case of accepted alternative in prediction, we replace it with the GT
    if unstress(target_phones) in target_accepted_alternatives:
        if (
            unstress(phonetic_detection)
            in target_accepted_alternatives[unstress(target_phones)]
        ):
            phonetic_detection = unstress(target_phones)
            g_d = g_t

    # print(phonetic_detection)

    # phonetic detection needs to be the stressed version for backwards compatibility (when correct, when it's not, we don't care)
    # print("phonetic_detection",phonetic_detection)
    # print("target_phones",target_phones)
    if phonetic_detection == unstress(target_phones):
        phonetic_detection = target_phones
        g_d = g_t
    return {
        "status": "success",
        "phonetic_detection": phonetic_detection,
        "gibberish_truth": '_'.join(g_t),
        "gibberish_detected": '_'.join(g_d),
    }


def phonemeContrast_from_formatted_phonetics_audio(
    audio,
    phonetics='T_ER1_N_D ER0|AW1_N_D',
    target_word_idx=0,
    target_syllable_idx=0,
    target_occurence_idx=0,  # will be 0  all the time for vowels, and most of the time for consonants
    target_phones='ER1',
    alternatives=cmu_vowels,
    max_speech_rate=8,
    mode: AudioMode = AudioMode.FILE,
    model=default_model,
    to_gibberish=cmu_to_gibberish,
    **kwargs,
):
    phonetics = phonetics.replace('CH', 'T_SH').replace('JH', 'D_ZH')
    g_t = [
        to_gibberish[unstress(p)]
        for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]
    ]
    audio_load, phone_prob_matrix = model.audio_to_phone_prob_matrix(
        audio, phonetics, max_speech_rate=max_speech_rate, mode=mode
    )

    if audio_load.status != AudioStatus.SUCCESS:
        return {
            "status": str(audio_load.status),
            "phonetic_detection": "null",
            "gibberish_truth": '_'.join(g_t),
            "gibberish_detected": "null",
        }
    phoneme_list = phonetics.replace(' ', '_').replace('|', '_').split('_')
    df_segmented, _ = model.phone_prob_matrix_segmentation(
        phone_prob_matrix, remove_stress_annots(phoneme_list)
    )

    # if the probabilities were too low,
    # maybe change this to empty if we want the other feedback "are you saying the right words",
    # or change null to "non-speech", nothing, nonsense or the pred_phones_audio
    if model.status != AudioStatus.SUCCESS:
        # convert to gibberish, but translate UNK token to 'uh', the schwa because we don't know what it is
        g_d = [
            to_gibberish[unstress(p)] if not 'UNK' in p else 'uh'
            for p in model.pred_phones_audio
        ]
        if g_d == []:
            return {
                "status": model.status,
                "phonetic_detection": "null",
                "gibberish_truth": '_'.join(g_t),
                "gibberish_detected": 'nothing',
            }
        else:
            if len(g_d) > 10:
                return {
                    "status": model.status,
                    "phonetic_detection": "null",
                    "gibberish_truth": '_'.join(g_t),
                    "gibberish_detected": 'nonsense',
                }
            else:
                return {
                    "status": model.status,
                    "phonetic_detection": "null",
                    "gibberish_truth": '_'.join(g_t),
                    "gibberish_detected": '_'.join(g_d),
                }

    return phonemeContrast_from_df_segmented(
        df_segmented,
        phonetics=phonetics,
        target_word_idx=target_word_idx,
        target_syllable_idx=target_syllable_idx,
        target_occurence_idx=target_occurence_idx,
        target_phones=target_phones,
        alternatives=alternatives,
        model=model,
        to_gibberish=to_gibberish,
    )


def schwa_sound_from_formatted_phonetics_audio(
    audio,
    phonetics="AY1 W_UH1_D L_AH1_V T_UW1 G_OW1 T_UW1 AY1|ER0|L_AH0_N_D",
    vowels=cmu_stressed_vowels,
    target_word_idx=0,
    target_syllable_idx=0,
    target_occurence_idx=0,  # will be 0  all the time for vowels, and most of the time for consonants
    target_phones='AH0',
    # alternatives=cmu_vowels,
    alternatives=cmu_stressed_vowels,
    max_speech_rate=8,
    mode: AudioMode = AudioMode.FILE,
    # model=default_model_stressed,
    model=default_model,
    to_gibberish=cmu_to_gibberish,
    **kwargs,
):
    phonetics = phonetics.replace('CH', 'T_SH').replace('JH', 'D_ZH')
    g_t = [
        to_gibberish[unstress(p)]
        for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]
    ]
    audio_load, phone_prob_matrix = model.audio_to_phone_prob_matrix(
        audio, phonetics, max_speech_rate=max_speech_rate, mode=mode
    )
    if audio_load.status != AudioStatus.SUCCESS:
        return {
            "status": audio_load.status,
            "phonetic_detection": "null",
            "gibberish_truth": '_'.join(g_t),
            "gibberish_detected": "null",
        }
    phoneme_list = phonetics.replace(' ', '_').replace('|', '_').split('_')
    df_segmented, _ = model.phone_prob_matrix_segmentation(
        phone_prob_matrix, remove_stress_annots(phoneme_list)
    )

    # if the probabilities were too low,
    # maybe change this to empty if we want the other feedback "are you saying the right words",
    # or change null to "non-speech", nothing, nonsense or the pred_phones_audio
    if model.status != "success":
        # convert to gibberish, but translate UNK token to 'uh', the schwa because we don't know what it is
        g_d = [
            to_gibberish[unstress(p)] if not 'UNK' in p else 'uh'
            for p in model.pred_phones_audio
        ]
        if g_d == []:
            return {
                "status": model.status,
                "phonetic_detection": "null",
                "gibberish_truth": '_'.join(g_t),
                "gibberish_detected": 'nothing',
            }
        else:
            if len(g_d) > 10:
                return {
                    "status": model.status,
                    "phonetic_detection": "null",
                    "gibberish_truth": '_'.join(g_t),
                    "gibberish_detected": 'nonsense',
                }
            else:
                return {
                    "status": model.status,
                    "phonetic_detection": "null",
                    "gibberish_truth": '_'.join(g_t),
                    "gibberish_detected": '_'.join(g_d),
                }

    stress_result = stress_from_df_segmented_audio(
        audio_load.s,
        df_segmented,
        phonetics=phonetics,
        level="word",
        model=model,
        vowels=vowels,
    )
    # {"status": "success", "stress_intensities": word_intensities, "stress_binaries": word_bins}
    phoneme_contrast_dict = phonemeContrast_from_df_segmented(
        df_segmented,
        phonetics=phonetics,
        target_word_idx=target_word_idx,
        target_syllable_idx=target_syllable_idx,
        target_occurence_idx=target_occurence_idx,
        target_phones=target_phones,
        alternatives=alternatives,
        model=model,
        to_gibberish=to_gibberish,
    )
    # {"status": "success", "phonetic_detection": phonetic_detection, "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}

    if stress_result.satus == "success":
        return {"status": stress_result.status}
    elif phoneme_contrast_dict['status'] != "success":
        return {"status": phoneme_contrast_dict['status']}
    else:
        stress_intensities_word = stress_result.stress_intensities[target_word_idx]

        # find the least stressed vowels according to a ranking and threshold
        # if threshold=0, no contraint due to threshold, only ranking constraint
        unstressed_bin_f = lambda stress_intensities_word: intensity_to_bin(
            [100 - el for el in stress_intensities_word],
            n_max=roundup(len(stress_intensities_word) / 3),
            threshold=0,
        )
        unstressed_bin = unstressed_bin_f(stress_intensities_word)
        detected_target_vowel = phoneme_contrast_dict['phonetic_detection']

        if (
            unstressed_bin[target_syllable_idx] == 1
            and detected_target_vowel == target_phones
        ):
            is_schwa = True
        else:
            is_schwa = False
        return {
            "status": "success",
            "is_schwa": is_schwa,
            "unstressed_bin": unstressed_bin,
            "stress_intensities_word": stress_intensities_word,
            "target_vowel": detected_target_vowel,
        }


target_to_basis = {
    "HH": "HH",
    '': 'HH',
    'D': 'IH0_D',
    'IH0_D': 'IH0_D',
    'T': 'IH0_T',
    'S': 'IH0_S',
    'Z': 'IH0_Z',
    'IH0_Z': 'IH0_Z',
}


def phonetic_reference_processing(
    phonetics,
    target_word_idx,
    target_syllable_idx,
    target_phones,
    basis,
    position,
    contrast_idx,
):
    ### Phonetics processing to build a reference with the basis replacing target_phones
    phonetics = phonetics.replace('-', ' ').replace('{', '').replace('}', '')

    phonetics_indexed_df = phonetics_indexed_df_from_formatted_phonetics(
        phonetics.split(' ')[target_word_idx]
    )

    syl_GT = remove_stress_annots(
        phonetics.split(' ')[target_word_idx].split('|')[target_syllable_idx].split('_')
    )
    split_phonetics_words = [p.replace('|', '_').split('_') for p in phonetics.split(' ')]
    split_phonetics_syls = [
        [syl.split('_') for syl in p.split('|')] for p in phonetics.split(' ')
    ]

    # I cannot only split with "_" because splitting ''  with "_" generate a list with 1 element: [''],  and its len is 1.
    # So I filter these
    n_p_target = len([el for el in target_phones.split('_') if el != ''])
    n_ter_basis = len([el for el in basis.split('_') if el != ''])

    syl_idxs = phonetics_indexed_df.syl_idx.tolist()

    if position == "end":
        # in the target syllable, replace the barget with the basis
        split_phonetics_syls[target_word_idx][target_syllable_idx] = split_phonetics_syls[
            target_word_idx
        ][target_syllable_idx][:-n_p_target] + basis.split('_')
        # in the split phonetics at word level, just replace the word with the new version by merging the syllables
        split_phonetics_words[target_word_idx] = sum(
            split_phonetics_syls[target_word_idx], []
        )
        GT = syl_GT[-n_p_target - 1 :]
        syl_idxs = syl_idxs[:-n_p_target] + [syl_idxs[contrast_idx]] * n_ter_basis
    elif position == "start":
        # in the target syllable, replace the barget with the basis
        split_phonetics_syls[target_word_idx][target_syllable_idx] = [
            el for el in basis.split('_') if el != ''
        ] + split_phonetics_syls[target_word_idx][target_syllable_idx][n_p_target:]

        # split_phonetics_words[target_word_idx]=basis.split('_')+target_word[n_p_target:]
        split_phonetics_words[target_word_idx] = sum(
            split_phonetics_syls[target_word_idx], []
        )

        GT = syl_GT[: n_p_target + 1]
        syl_idxs = [syl_idxs[contrast_idx]] * n_ter_basis + syl_idxs[n_p_target:]

    return split_phonetics_words, syl_GT, GT, syl_idxs


from Bio import pairwise2

from scripts.mfa_utils import unicode_chars_to_words, words_to_unicode_chars


def list_pairwise_alignment(LISTA, LISTB):
    """maps list elements to unicode characters to be able to use the character base pariwise alignment from bipython,
    then map back results to list elements

    source: https://www.biostars.org/p/246408/
    """

    LISTA__, LISTB__, LATtoHAN, HANtoLAT = words_to_unicode_chars(LISTA, LISTB)
    alignments = pairwise2.align.globalxx(LISTA__, LISTB__)
    if len(alignments) == 0:
        return None, None
    res_a, res_b = unicode_chars_to_words(
        alignments[0].seqA, alignments[0].seqB, LATtoHAN, HANtoLAT
    )
    return res_a, res_b


@dataclass
class PostAnalysisResult:
    df_segmented: pd.DataFrame | None
    per_aligned: float
    dtw_cost: float
    silent_frame_ratio: float
    phone_count_ratio: float
    predicted_phones: list[str] | None
    predicted_aligned_phones: list[str] | None
    expected_aligned_phones: list[str] | None


def post_analysis(
    phone_prob_matrix: np.ndarray,
    phonetics: str,
    model: Wav2Vec2ForFramePrediction = default_model,
    proba_thresh: float = 0.5,
) -> PostAnalysisResult | None:
    """Verify matching between predicted phones and expected phones.
    Useful for rejecting recordings which do not match the assignment.

    Parameters
    ----------
    phone_prob_matrix : np.ndarray
        Phone probability matrix from the model
    phonetics : str
        Expected phonetics (word or sentence)
    model : Wav2Vec2ForFramePrediction, optional
        Model to use, by default default_model
    proba_thresh : float, optional
        Probability threshold for phone prediction, by default 0.5

    Returns
    -------
    PostAnalysisResults: summary of the quality checks
    """
    split_phonetics = [p.replace('|', '_').split('_') for p in phonetics.split(' ')]
    expected_phoneme_list = remove_stress_annots(sum(split_phonetics, []))

    (
        max_posterior_df,
        max_posterior_df_filtered_processed,
        max_posterior_df_filtered_processed_threshed,
    ) = model.max_posterior_phone_df(phone_prob_matrix, proba_thresh=proba_thresh)

    pred_phoneme_list = max_posterior_df_filtered_processed_threshed.phone.tolist()
    silent_frame_ratio = (max_posterior_df.phone == '[SIL]').mean()
    phone_count_ratio = len(max_posterior_df_filtered_processed_threshed) / len(
        expected_phoneme_list
    )

    pred_aligned, exp_aligned = list_pairwise_alignment(
        pred_phoneme_list, expected_phoneme_list
    )
    if not (pred_aligned and exp_aligned):
        return None

    pairwise_align_df = pd.DataFrame([exp_aligned, pred_aligned])
    pairwise_align_df.index = ['expected', 'predicted']

    per_aligned = levenshtein_distance(exp_aligned, pred_aligned) / len(exp_aligned)
    # print(f"expected: {RESB}, predicted: {RESA}, error_rate: {error_rate}, per: {per_aligned}")

    # Perform DTW between expected phones and predicted probability matrix
    df_segmented, dtw_cost = model.phone_prob_matrix_segmentation(
        phone_prob_matrix, expected_phoneme_list
    )

    # DTW failed
    if len(df_segmented) == 0:
        return PostAnalysisResult(
            None,
            per_aligned,
            None,
            silent_frame_ratio,
            phone_count_ratio,
            pred_phoneme_list,
            pred_aligned,
            exp_aligned,
        )

    dtw_cost = -dtw_cost / len(pred_phoneme_list)
    df_segmented['n_frames'] = df_segmented.end_idx - df_segmented.start_idx
    # insert a row at the dashes position by using float indexing
    dash_indexes = pairwise_align_df.T[pairwise_align_df.T.expected == '-'].index.tolist()
    for dash_idx in dash_indexes:
        df_segmented.loc[dash_idx - 0.5] = np.nan
        df_segmented = df_segmented.sort_index()
        df_segmented = df_segmented.reset_index(drop=True)

    df_segmented['predicted_phones'] = pairwise_align_df.T.predicted
    df_segmented = df_segmented.reset_index(drop=True)

    return PostAnalysisResult(
        df_segmented,
        per_aligned,
        dtw_cost,
        silent_frame_ratio,
        phone_count_ratio,
        pred_phoneme_list,
        pred_aligned,
        exp_aligned,
    )


def start_end_contrast_from_prob_matrix(
    phone_prob_matrix,
    phonetics='T_ER1_N_D ER0|AW1_N_D',
    target_word_idx=0,
    target_syllable_idx=None,
    target_phones='D',
    # basis='[UNK]_D',
    basis=None,
    max_speech_rate=8,
    mode: AudioMode = AudioMode.FILE,
    position="end",
    model=default_model,
    vowels=cmu_vowels,
    consonants=cmu_consonants,
    to_gibberish=cmu_to_gibberish,
    **kwargs,
):
    g_t = [
        to_gibberish[unstress(p)]
        for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]
    ]
    if position == "end":
        contrast_idx = -1
    elif position == "start":
        contrast_idx = 0
    else:
        print("contrast should be start or end")

    if basis is None:
        basis = target_to_basis[target_phones]
    split_phonetics_by_words, syl_GT, GT, syl_idxs = phonetic_reference_processing(
        phonetics,
        target_word_idx,
        target_syllable_idx,
        target_phones,
        basis,
        position,
        contrast_idx,
    )

    phoneme_list = sum(split_phonetics_by_words, [])
    df_segmented, _ = model.phone_prob_matrix_segmentation(
        phone_prob_matrix, remove_stress_annots(phoneme_list)
    )
    df_word = extract_word(df_segmented, split_phonetics_by_words, target_word_idx)

    # FIXME extremely bad style
    if model.status != "success":
        # g_d=[to_gibberish[unstress(p)] for p in model.pred_phones_audio]
        g_d = [
            to_gibberish[unstress(p)] if not 'UNK' in p else 'uh'
            for p in model.pred_phones_audio
        ]

        if g_d == []:
            return {
                "status": model.status,
                "phonetic_detection": "null",
                "gibberish_truth": '_'.join(g_t),
                "gibberish_detected": 'nothing',
            }
        else:
            if len(g_d) > 10:
                return {
                    "status": model.status,
                    "phonetic_detection": "null",
                    "gibberish_truth": '_'.join(g_t),
                    "gibberish_detected": 'nonsense',
                }
            else:
                return {
                    "status": model.status,
                    "phonetic_detection": "null",
                    "gibberish_truth": '_'.join(g_t),
                    "gibberish_detected": '_'.join(g_d),
                }

    # selection of the syllable, then divide it into a root and a termination (or start and root)
    df_word['syl_idx'] = syl_idxs

    # As we use index based on numbers in dataframe, cyclical indexing is not completely supported. I just add the case here for the use of -1
    if target_syllable_idx == -1:
        target_syllable_idx = syl_idxs[-1]
    df_syl = df_word[df_word.syl_idx == target_syllable_idx]
    df_syl = df_syl[df_syl.pred_phones_audio != '[SIL]']

    # filter out phonemes too short inside the termination,
    # if forced alignment lead to assigning very few frames for a phoneme, we assume it means it does not really exists
    # the minimal duration end-start difference is 0.02 (equals 1 frame), hence this filter:
    df_syl = df_syl[(df_syl.end - df_syl.start) > model.time_per_output]

    n_ter_basis = len([el for el in basis.split('_') if el != ''])
    #  In this case (e.g. starting h-) we want to use the fact that it is possible to reduce the set of possibilities to vowels or consonants as there is only 1 phoneme.
    # It's the same method as in vowel/consonant contrast
    if n_ter_basis == 1 and basis != '':
        p_idx = contrast_idx
        if unstress(basis) in consonants:
            phoneme_set = consonants
        elif unstress(basis) in vowels:
            phoneme_set = vowels

        phoneme_set_ids = [
            model.p_to_id[el] for el in remove_stress_annots(phoneme_set)
        ] + ["[SIL]"]

        if len(df_syl) > 0:
            proba_means = df_syl.iloc[p_idx].proba_means

            # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
            filtered_proba_means = [
                0 if i not in phoneme_set_ids else el for i, el in enumerate(proba_means)
            ]
            idx_mean_max = np.argmax(filtered_proba_means)
            phonetic_detection = model.id_to_p[int(idx_mean_max)]

            # As we use index based on numbers in dataframe, cyclical indexing can be supported by taking the the last df index from the list
            df_syl.loc[df_syl.index[p_idx], "pred_phones_audio"] = phonetic_detection

    # syl_detected=drop_consecutive_duplicates(df_syl[['pred_phones_audio']]).pred_phones_audio.tolist()
    syl_detected = df_syl[['pred_phones_audio']].pred_phones_audio.tolist()

    if position == "end":
        ter = syl_detected[-n_ter_basis - 1 :]
    elif position == "start":
        # detected start
        ter = syl_detected[: n_ter_basis + 1]

    ter = drop_consecutive_duplicate_elements(ter)

    if len(ter) == len(GT):
        ter_post = []
        for i, p in enumerate(ter):
            # post-correction: for the target, when we are in a case of accepted alternative in prediction, we replace it with the GT
            if unstress(GT[i]) in target_accepted_alternatives:
                if unstress(p) in target_accepted_alternatives[unstress(GT[i])]:
                    ter_post.append(unstress(GT[i]))
                else:
                    ter_post.append(unstress(p))
            else:
                ter_post.append(unstress(p))
    else:
        ter_post = ter

    # there might be consecutive duplicates when we concatenate root and ter_post
    # g_d=drop_consecutive_duplicate_elements([to_gibberish[unstress(p)] for p in root+ter_post])

    # convert to gibberish, but translate UNK token to 'uh', the schwa because we don't know what it is
    g_d = [to_gibberish[unstress(p)] if not 'UNK' in p else 'uh' for p in ter_post]

    if position == "end":
        detection = ter_post[1:]
    elif position == "start":
        detection = ter_post[:-1]

    target_phones_list = [el for el in target_phones.split('_') if el != '']

    # for words like "ordered", at this step detection would be ['ER', 'D'], and won't be in terminations_accepted_alternatives of "D"
    # because it is an accepted alternative of "IH_D"
    if len(detection) > 0:
        if position == "end":
            if detection == syl_GT[-len(detection) :]:
                detection = target_phones_list
        elif position == "start":
            if detection == syl_GT[: -len(detection)]:
                detection = target_phones_list

    detection = '_'.join(detection)

    # phonetic detection needs to be the stressed version for backwards compatibility
    # however, here I convert to stressed version only when correct
    if (
        target_phones != ''
    ):  # don't try to split in case of case of final -s "nothing" target
        # if dectection is the same as target_phones (without the stress marks because charsiu don't put that), change back to target phones
        if detection.split('_') == remove_stress_annots(target_phones_list):
            detection = target_phones

        # post-correction for the whole termination not to differentiate between IH_D and AH_D     or    IH_Z and AH_Z
        unstressed_target = '_'.join(remove_stress_annots(target_phones_list))
        if unstressed_target in terminations_accepted_alternatives:
            if detection in terminations_accepted_alternatives[unstressed_target]:
                detection = target_phones

    return {
        "status": "success",
        "phonetic_detection": detection,
        "gibberish_truth": '_'.join(g_t),
        "gibberish_detected": '_'.join(g_d),
    }


def start_end_contrast_from_formatted_phonetics_audio(
    audio,
    phonetics='T_ER1_N_D ER0|AW1_N_D',
    target_word_idx=0,
    target_syllable_idx=None,
    target_phones='D',
    # basis='[UNK]_D',
    basis=None,
    max_speech_rate=8,
    mode: AudioMode = AudioMode.FILE,
    position="end",
    model=default_model,
    vowels=cmu_vowels,
    consonants=cmu_consonants,
    to_gibberish=cmu_to_gibberish,
    **kwargs,
):
    # by default we apply the logic at the word level. End of word means last syllable, start of word means first syllable
    if target_syllable_idx is None:
        if position == "end":
            target_syllable_idx = -1
        elif position == "start":
            target_syllable_idx = 0
        else:
            print("position should be start or end")

    g_t = [
        to_gibberish[unstress(p)]
        for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]
    ]
    audio_load, phone_prob_matrix = model.audio_to_phone_prob_matrix(
        audio, phonetics, max_speech_rate=max_speech_rate, mode=mode
    )
    if audio_load.status != AudioStatus.SUCCESS:
        return {
            "status": audio_load.status,
            "phonetic_detection": "null",
            "gibberish_truth": '_'.join(g_t),
            "gibberish_detected": "null",
        }

    return start_end_contrast_from_prob_matrix(
        phone_prob_matrix,
        phonetics=phonetics,
        target_word_idx=target_word_idx,
        target_syllable_idx=target_syllable_idx,
        target_phones=target_phones,
        basis=basis,
        max_speech_rate=max_speech_rate,
        mode=mode,
        position=position,
        model=model,
        vowels=vowels,
        consonants=consonants,
        to_gibberish=to_gibberish,
    )


def phonetic_content_analysis(
    s, phonetics, model=default_model, vowels=cmu_vowels, consonants=cmu_consonants
):
    phonetic_content = model.analyze_phonetic_content(s, phonetics)
    if len(phonetic_content) == 0:
        return phonetic_content

    def syl_analysis(syl_df):
        # inside a syllable or word, there cannot be several times the same phoneme consecutively
        collapsed_syl = drop_consecutive_duplicates(syl_df[['pred_phones_audio']])

        # one syllable in ground truth can correspond in several syllables in prediction, e.g. moved -> movED
        # or also in "0 syllable" if there is no vowel. If that's the case,  I have to consider it is 1 syllable
        pred_syls = SonoriPy(collapsed_syl.pred_phones_audio.tolist())[0]
        if pred_syls == []:
            pred_syls = [collapsed_syl.pred_phones_audio.tolist()]

        # extract syllable indices for predicted syls
        pred_syls_indxs = sum(
            [[i] * n for i, n in enumerate([len(syl) for syl in pred_syls])], []
        )

        collapsed_syl.loc[:, 'pred_syls_indxs_inside_GT_syl'] = pred_syls_indxs

        # I align the collapsed syllable to the timed one. This leads to NaNs that have to be filled
        syl_df.loc[:, 'pred_syls_indxs_inside_GT_syl'] = collapsed_syl.loc[
            :, 'pred_syls_indxs_inside_GT_syl'
        ].astype(int)
        syl_df = syl_df.fillna(method="ffill")

        # in each syllable in prediction, I only keep one vowel, by majority vote, i.e. I drop all vowels except max frames in each syl
        # for i in syl_df.pred_syls_indxs_inside_GT_syl.unique():
        #     s=syl_df.loc[syl_df.pred_syls_indxs_inside_GT_syl==i]
        #     v=s[s.pred_phones_audio.isin(vowels)]
        #     if len(v)>0:
        #         m=v.n_frames.idxmax()
        #         l=[el for el in v.index.tolist() if el != m]
        #         syl_df=syl_df.drop(l)

        return syl_df

    # reduction: we go in each syllable
    dfs = []
    for w_idx in range(phonetic_content.word_idx.values[-1] + 1):
        w_df = phonetic_content[phonetic_content.word_idx == w_idx]
        for s_idx in range(w_df.syl_idx.values[-1] + 1):
            s_df = w_df[phonetic_content.syl_idx == s_idx]
            syl_df = syl_analysis(s_df)
            dfs.append(syl_df)
    phonetic_content = pd.concat(dfs)

    # post-correction : for each row when we are in a case of accepted alternative in prediction, we replace it with the GT
    for i, r in phonetic_content.iterrows():
        if r.phones in target_accepted_alternatives:
            if r.pred_phones_audio in target_accepted_alternatives[r.phones]:
                phonetic_content.loc[i, 'pred_phones_audio'] = r.phones

    # phonetic_content=phonetic_content[phonetic_content.n_frames>1]
    phonetic_content = phonetic_content[phonetic_content.pred_phones_audio != '[SIL]']
    phonetic_content = phonetic_content.loc[
        drop_consecutive_duplicates(
            phonetic_content[['phones', 'pred_phones_audio']]
        ).index,
        :,
    ]
    return phonetic_content


def analyze_start_end_for_synth_word(
    word,
    words_selected_df,
    target_word_idx=0,
    target_syllable_idx=-1,
    target_phones='Z',
    basis=None,
    position='end',
    model=default_model,
):
    from src.audio_processing import read_audio_file
    from src.text_processing import prefill_for_sentence

    phonetics = prefill_for_sentence(word)['phonetics']

    results = []
    for i, r in words_selected_df.iterrows():
        audio, fs = read_audio_file(r.path, fs=16000)
        res = start_end_contrast_from_formatted_phonetics_audio(
            audio,
            phonetics=phonetics,
            target_word_idx=target_word_idx,
            target_syllable_idx=target_syllable_idx,
            target_phones=target_phones,
            basis=basis,
            position=position,
            mode=AudioMode.NUMPY,
            model=model,
        )
        results.append(res)

    return results


def multiple_aspect_from_prob_matrix(
    phone_prob_matrix,
    s,
    phonetics="AY1 W_UH1_D L_AH1_V T_UW1 G_OW1 T_UW1 AY1|ER0|L_AH0_N_D",
    vowels=cmu_vowels,
    max_speech_rate=8,
    mode: AudioMode = AudioMode.FILE,
    model=default_model,
    to_gibberish=cmu_to_gibberish,
    **kwargs,
):
    phoneme_list = phonetics.replace(' ', '_').replace('|', '_').split('_')
    df_segmented, _ = model.phone_prob_matrix_segmentation(
        phone_prob_matrix, remove_stress_annots(phoneme_list)
    )
    # print('multi_aspect: df_segmented computed')

    stress_result = stress_from_df_segmented_audio(
        s,
        df_segmented,
        phonetics=phonetics,
        level=StressCategory.WORD,
        model=model,
        vowels=vowels,
    )
    # print('multi_aspect: stress_dict word computed')

    # in the speech tech through API, the phonetics is sent by chunk (list of strings) at flask level,
    # and I build the n_words_by_chunk and re-join the phonetics as 1 string
    # Here I always assume a single chunk for now. I could use punctuation in text to compute n_words_by_chunk as well
    n_words_by_chunk = [len(phonetics.split(' '))]
    stress_result_sentence = stress_from_df_segmented_audio(
        s,
        df_segmented,
        phonetics=phonetics,
        n_words_by_chunk=n_words_by_chunk,
        level=StressCategory.SENTENCE,
        model=model,
        vowels=vowels,
    )
    # print('multi_aspect: stress_dict_sentence computed')

    phonetics_indexed_df = phonetics_indexed_df_from_formatted_phonetics(phonetics)
    is_vowel = phonetics_indexed_df.apply(lambda r: unstress(r.phones) in vowels, axis=1)
    phonetics_indexed_df['is_vowel'] = is_vowel
    vowels_indexed_df = phonetics_indexed_df[is_vowel].copy()
    # print('multi_aspect: indexes computed')

    clusters = []
    for w_i, w_df in phonetics_indexed_df.groupby('word_idx'):
        for s_i, s_df in w_df.groupby('syl_idx'):
            initial_cluster = s_df[s_df.is_vowel.cumsum() == 0]
            final_cluster = s_df[s_df.is_vowel.cumsum() == 1].iloc[1:]
            clusters.append(
                {
                    'cluster': '_'.join(initial_cluster.phones.tolist()),
                    'word_idx': w_i,
                    'syl_idx': s_i,
                    'position': 'start',
                }
            )
            clusters.append(
                {
                    'cluster': '_'.join(final_cluster.phones.tolist()),
                    'word_idx': w_i,
                    'syl_idx': s_i,
                    'position': 'end',
                }
            )
    clusters_df = pd.DataFrame(clusters)
    # print('multi_aspect: clusters_df computed')

    # performing vowel contrasts based on the vowels_indexed_df
    vowel_detections = []
    for i, r in vowels_indexed_df.iterrows():
        phoneme_contrast_dict = phonemeContrast_from_df_segmented(
            df_segmented,
            phonetics=phonetics,
            target_word_idx=r.word_idx,
            target_syllable_idx=r.syl_idx,
            target_occurence_idx=0,
            target_phones=r.phones,
            alternatives=vowels,
            model=model,
            to_gibberish=to_gibberish,
        )
        vowel_detections.append(phoneme_contrast_dict)
        # {"status": "success", "phonetic_detection": phonetic_detection, "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}
    vowel_detections_df = pd.DataFrame(vowel_detections)
    vowels_indexed_df['detection'] = vowel_detections_df.phonetic_detection.tolist()
    # print('multi_aspect: vowels_indexed_df computed')

    cluster_detections = []
    for i, r in clusters_df.iterrows():
        # target_phones = r.cluster if r.cluster != '' else 'P'
        # basis = r.cluster if r.cluster != '' else 'P'
        if r.cluster != '':
            res = start_end_contrast_from_prob_matrix(
                phone_prob_matrix,
                phonetics=phonetics,
                target_word_idx=r.word_idx,
                target_syllable_idx=r.syl_idx,
                target_phones=r.cluster,
                basis=r.cluster,
                position=r.position,
                mode=mode,
                model=model,
            )
            cluster_detections.append(res)
    cluster_detections_df = pd.DataFrame(cluster_detections)
    # print('multi_aspect: vowels_indexed_df computed')

    clusters_df = clusters_df[clusters_df.cluster != '']
    clusters_df['detection'] = cluster_detections_df.phonetic_detection.tolist()

    vowels_indexed_df = vowels_indexed_df[["phones", "word_idx", "syl_idx", "detection"]]
    vowels_indexed_df['position'] = "middle"
    vowels_indexed_df = vowels_indexed_df[
        ["phones", "word_idx", "syl_idx", "position", "detection"]
    ]

    vowels_indexed_df['stress_intensities'] = sum(stress_result.stress_intensities, [])
    vowels_indexed_df['stress_binaries_word'] = sum(stress_result.stress_binaries, [])
    clusters_df.columns = ["phones", "word_idx", "syl_idx", "position", "detection"]

    merged_results = pd.concat([vowels_indexed_df, clusters_df])

    syl_dfs = []
    for w_i, w_df in merged_results.groupby('word_idx'):
        w_df['stress_binaries_sentence'] = stress_result_sentence.stress_binaries[w_i]
        for s_i, s_df in w_df.groupby('syl_idx'):
            map_to_i = {'start': 0, 'middle': 1, 'end': 2}
            s_df['position_idx'] = s_df['position'].apply(lambda r: map_to_i[r])
            # repeat the vallue of the nucleus on the whole syllable, not to have nans
            s_df['stress_intensities'] = s_df['stress_intensities'].dropna().values[0]
            s_df['stress_binaries_word'] = s_df['stress_binaries_word'].dropna().values[0]

            syl_dfs.append(s_df.sort_values('position_idx'))
    final_merged_results = pd.concat(syl_dfs)

    return final_merged_results


def multiple_aspect_from_formatted_phonetics_audio(
    audio: AudioInput,
    phonetics: str,
    vowels: set = cmu_vowels,
    max_speech_rate: int = 8,
    mode: AudioMode = AudioMode.FILE,
    model: Wav2Vec2ForFramePrediction = default_model,
    to_gibberish: dict[str, str] = cmu_to_gibberish,
) -> tuple[AudioLoadResult, pd.DataFrame | None, PostAnalysisResult | None]:
    phonetics = cmu_ensure_phonetics_consistency(phonetics)
    audio_load, phone_prob_matrix = model.audio_to_phone_prob_matrix(
        audio, phonetics, max_speech_rate=max_speech_rate, mode=mode
    )

    if audio_load.status != AudioStatus.SUCCESS:
        return audio_load, None, None

    # this operation is done in audio_to_phone_prob_matrix, but I have to do it again here then for consistency
    phonetics = phonetics.replace('-', ' ').replace('{', '').replace('}', '')

    post_analysis_result = post_analysis(phone_prob_matrix, phonetics, model=model)

    return (
        audio_load,
        multiple_aspect_from_prob_matrix(
            phone_prob_matrix,
            audio_load.waveform,
            phonetics=phonetics,
            vowels=vowels,
            max_speech_rate=max_speech_rate,
            mode=mode,
            model=model,
            to_gibberish=to_gibberish,
        ),
        post_analysis_result,
    )


def use_tests():
    # from DL_speech_tech import *

    from src.audio_processing import read_audio_file
    from src.text_processing import prefill_for_sentence, remove_stress_annots
    from src.label_data_processing import actor_recordings, synth_words_data

    default_model_ipa = Wav2Vec2ForFramePrediction(ipa_alphabet, w2v2_model_format="onnx")
    default_model_ipa.load(name='model_mailabs_pca_95_knn_10_w_ipa')
    default_model_stressed = Wav2Vec2ForFramePrediction(
        cmu_stressed_alphabet, w2v2_model_format="onnx"
    )
    default_model_stressed.load(
        name='model_mailabs_equilibrated_stressed_pca_95_knn_10_cos_w'
    )

    from src.label_data_processing import (
        actor_recordings,
        final_s_artificial_data,
        synth_words_data,
    )

    df_words = synth_words_data().dropna()

    word = "perhaps"
    words_selected_df = df_words[df_words.text == word]
    analyze_start_end_for_synth_word(
        word,
        words_selected_df,
        target_word_idx=0,
        target_syllable_idx=1,
        target_phones='HH',
        position='start',
        model=default_model,
    )

    word = "ability"
    phonetics = prefill_for_sentence(word)['phonetics']
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_" + word + ".mp3"
    audio, fs = read_audio_file(path, fs=16000)
    target_phones = "AH0"
    target_syllable_idx = [
        1 if target_phones in el else 0 for el in phonetics.split('|')
    ].index(1)
    schwa_sound_from_formatted_phonetics_audio(
        audio,
        phonetics=phonetics,
        min_stress=0,
        max_stress=20,
        vowels=cmu_vowels,
        target_word_idx=0,
        target_syllable_idx=target_syllable_idx,
        target_phones='AH0',
        alternatives=cmu_vowels,
        mode='numpy',
        model=default_model,
        to_gibberish=cmu_to_gibberish,
    )

    from src.pronunciation_dictionaries import get_augmented_mfa_dict, mfa_to_display_ipa

    mfa_gb = get_augmented_mfa_dict("en_GB")
    mfa_us = get_augmented_mfa_dict("en_US")
    word = "history"
    phonetics = prefill_for_sentence(
        word, lang="en_US", mode="MFA_IPA", word_dict=mfa_us
    )['phonetics']
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_" + word + ".mp3"
    s, fs = read_audio_file(path, fs=16000)
    phonemeContrast_from_formatted_phonetics_audio(
        s,
        phonetics=phonetics,
        target_word_idx=0,
        target_syllable_idx=1,
        target_phones='ə0',
        alternatives=ipa_vowels,
        to_gibberish=mfa_to_display_ipa,
        mode='numpy',
        model=default_model_ipa,
    )

    # SBL-0001_Aback_48k_mono.mp3
    regex = "SBL-[0-9]*_[A-Za-z]*_48k_mono.mp3"

    from glob import glob

    path = "/mnt/c/Users/noe_t/Downloads/TTS - Native English Speakers/*/*"
    paths = glob(path)
    target_phones = "AH0"

    import re
    import os

    records = []
    for p in paths:
        _, fn = os.path.split(p)

        if re.match(regex, fn):
            word = fn
            word = word.split('_')[1].lower()
            phonetics = prefill_for_sentence(word)['phonetics']

            # make sure there is actually a schwa in the word according to CMU, because apparently, these word were not selected by checking CMU phonemes
            # but maybe some dictionary online like oxford or so
            try:
                target_syllable_idx = [
                    1 if target_phones in el else 0 for el in phonetics.split('|')
                ].index(1)
            except:
                break
            audio, fs = read_audio_file(p, fs=16000)
            schwa_res = schwa_sound_from_formatted_phonetics_audio(
                audio,
                phonetics=phonetics,
                vowels=cmu_vowels,
                target_word_idx=0,
                target_syllable_idx=target_syllable_idx,
                target_phones='AH0',
                alternatives=cmu_vowels,
                mode='numpy',
                model=default_model,
                to_gibberish=cmu_to_gibberish,
            )

            # phonetics=prefill_for_sentence(word, lang="en_US", mode="MFA_IPA", word_dict=mfa_us)['phonetics']
            phone_res = phonemeContrast_from_formatted_phonetics_audio(
                audio,
                phonetics=prefill_for_sentence(
                    word, lang="en_US", mode="MFA_IPA", word_dict=mfa_us
                )['phonetics'],
                target_word_idx=0,
                target_syllable_idx=target_syllable_idx,
                target_phones='ə0',
                alternatives=ipa_vowels,
                to_gibberish=mfa_to_display_ipa,
                mode='numpy',
                model=default_model_ipa,
            )
            phone_res_stressed_cmu = phonemeContrast_from_formatted_phonetics_audio(
                audio,
                phonetics=phonetics,
                target_word_idx=0,
                target_syllable_idx=target_syllable_idx,
                target_phones='AH0',
                model=default_model_stressed,
                alternatives=cmu_stressed_vowels,
                vowels=cmu_stressed_vowels,
                mode='numpy',
            )
            stress_res = stress_from_formatted_phonetics(
                audio, phonetics=phonetics, level="word", max_speech_rate=8, mode='numpy'
            )
            print(word)
            print(phonetics)
            print(schwa_res)
            print(phone_res)
            print(stress_res)
            record = {
                "word": word,
                "phonetics": phonetics,
                "schwa_res": schwa_res,
                "phone_res": phone_res,
                "stress_res": stress_res,
            }
            records.append(record)

    pd.DataFrame(records)
    pd.DataFrame(pd.DataFrame(records).schwa_res.tolist())
    pd.DataFrame(pd.DataFrame(records).phone_res.tolist())
    pd.DataFrame(pd.DataFrame(records).stress_res.tolist())

    word = "hundred"
    phonetics = prefill_for_sentence(word)['phonetics']
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_" + word + ".mp3"
    audio, fs = read_audio_file(path, fs=16000)
    res = start_end_contrast_from_formatted_phonetics_audio(
        audio,
        phonetics=phonetics,
        target_word_idx=0,
        target_syllable_idx=0,
        target_phones='N',
        basis="N",
        position='start',
        mode='numpy',
        model=default_model,
    )

    word = "picked"
    phonetics = prefill_for_sentence(word)['phonetics']
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_" + word + ".mp3"
    audio, fs = read_audio_file(path, fs=16000)
    res = start_end_contrast_from_formatted_phonetics_audio(
        audio,
        phonetics=phonetics,
        target_word_idx=0,
        target_syllable_idx=0,
        target_phones='T',
        basis=target_to_basis['T'],
        position='end',
        mode='numpy',
        model=default_model,
    )

    word = "orders"
    phonetics = prefill_for_sentence(word)['phonetics']
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_" + word + ".mp3"
    audio, fs = read_audio_file(path, fs=16000)
    res = start_end_contrast_from_formatted_phonetics_audio(
        audio,
        phonetics=phonetics,
        target_word_idx=0,
        target_phones='Z',
        position='end',
        mode='numpy',
        model=default_model,
    )
    # res=start_end_contrast_from_formatted_phonetics_audio(audio,phonetics=phonetics,target_word_idx=0,target_phones='Z',position='end',mode='numpy',model=default_model_charsiu)

    word = "cleans"
    phonetics = prefill_for_sentence(word)['phonetics']
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_" + word + ".mp3"
    s, fs = read_audio_file(path, fs=16000)
    res = start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=phonetics,
        target_word_idx=0,
        target_phones='Z',
        position='end',
        mode='numpy',
        model=default_model,
    )

    phone_prob_df.max(axis=1)
    max_idxs = np.argmax(phone_prob_df, axis=1)
    phone_prob_df.argmax(axis=1)

    alphabet = default_model.alphabet + ['[SIL]']

    max_posterior_df = pd.DataFrame()
    max_posterior_df['phone'] = [alphabet[i] for i in max_idxs]
    max_posterior_df['proba'] = phone_prob_df.max(axis=1)
    max_posterior_df_filtered = max_posterior_df[max_posterior_df.proba > 0.7][
        max_posterior_df.phone != "[SIL]"
    ]
    max_posterior_df_filtered.sort_values('proba', ascending=False).drop_duplicates(
        'phone'
    ).sort_index()

    path = 'data/it.wav'
    audio, fs = read_audio_file(path, fs=16000)
    phone_prob_matrix = default_model.predict_phone_prob_matrix(s, default_model.fs)
    phoneme_list = 'IH1_T'.replace(' ', '_').replace('|', '_').split('_')
    df_segmented, _ = default_model.phone_prob_matrix_segmentation(
        phone_prob_matrix, remove_stress_annots(phoneme_list)
    )

    phone_prob_df = pd.DataFrame(phone_prob_matrix)
    phone_prob_df.columns = default_model.alphabet + ["[SIL]"]

    self = default_model_ipa
    phone_prob_matrix = self.predict_phone_prob_matrix(s, self.fs)
    phoneme_list = phonetics.replace(' ', '_').replace('|', '_').split('_')
    df_segmented, _ = default_model_ipa.phone_prob_matrix_segmentation(
        phone_prob_matrix, remove_stress_annots(phoneme_list)
    )

    path = 'data/synth_audio/cmu_words/standard/prosody/Brian/M_UK_ekk.mp3'
    formatted_phonetics = prefill_for_sentence('ekk')['phonetics']
    s, fs = read_audio_file(path, fs=16000)
    phonemeContrast_from_formatted_phonetics_audio(
        s,
        phonetics=formatted_phonetics,
        target_word_idx=0,
        target_syllable_idx=1,
        target_phones='EY1',
        alternatives=cmu_vowels,
        mode='numpy',
    )

    path = 'data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf'
    # path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a'
    # path='data/audio_recordings/turned_around.mp3'
    s, fs = read_audio_file(path, fs=16000)
    formatted_phonetics = prefill_for_sentence('I would love to go to ireland')[
        'phonetics'
    ]
    stress_from_formatted_phonetics(
        s,
        phonetics=formatted_phonetics,
        level="sentence",
        n_words_by_chunk=[7],
        max_speech_rate=8,
        mode='numpy',
    )

    # path='data/synth_audio/cmu_words/standard/prosody/Brian/M_UK_international.mp3'
    path = "data/synth_audio/cmu_words/test_Lea_french_accent/F_FR_annotation.mp3"
    s, fs = read_audio_file(path, fs=16000)
    formatted_phonetics = prefill_for_sentence('annotation')['phonetics']
    stress_from_formatted_phonetics(
        s,
        phonetics=formatted_phonetics,
        level="word",
        # n_words_by_chunk=[7],
        max_speech_rate=8,
        mode='numpy',
    )

    path = 'data/synth_audio/cmu_words/standard/prosody/Brian/M_UK_france.mp3'
    s, fs = read_audio_file(path, fs=16000)
    formatted_phonetics = prefill_for_sentence('france')['phonetics']
    stress_from_formatted_phonetics(
        s, phonetics=formatted_phonetics, level="word", mode='numpy'
    )

    df = synth_words_data().dropna()
    df['target_word_indexes'] = 0
    df['target_syllable_indexes'] = 0
    df['contrast'] = 'start'
    df['fpath'] = df['path']
    clusters = ["TH_R", "P_R", "S_P_L", "S_K_R"]
    cluster = clusters[-1]

    df.phonetics.str.startswith(cluster)
    df_c = df[df.phonetics.str.startswith(cluster)]

    row = df_c.iloc[0]
    s, fs = read_audio_file(row.path, fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=row.phonetics,
        target_word_idx=0,
        target_phones=cluster,
        basis=cluster,
        position='start',
        mode='numpy',
    )

    word = "listens"
    phonetics = prefill_for_sentence(word)['phonetics']
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_" + word + ".mp3"
    s, fs = read_audio_file(path, fs=16000)
    res = start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=phonetics,
        target_word_idx=0,
        target_phones='Z',
        position='end',
        mode='numpy',
        model=default_model,
    )

    cmu_phonetics = "L_IH1|S_AH0_N"
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_listen.mp3"
    s, fs = read_audio_file(path, fs=16000)
    res = start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=cmu_phonetics,
        target_word_idx=0,
        target_phones='Z',
        position='end',
        mode='numpy',
        model=default_model,
    )

    cmu_phonetics = "L_IH1|S_AH0_N_D"
    path = "data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_listen.mp3"
    s, fs = read_audio_file(path, fs=16000)
    res = start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=cmu_phonetics,
        target_word_idx=0,
        target_phones='D',
        position='end',
        mode='numpy',
        model=default_model,
    )

    df = synth_words_data()
    df['target_word_indexes'] = 0
    df['target_syllable_indexes'] = -1
    df['fpath'] = df['path']
    df = df.dropna()

    df_z = df[
        (
            ~(df.phonetics.str.endswith('AH0_Z') | df.phonetics.str.endswith('IH0_Z'))
            & ~df.phonetics.str.endswith('_S')
        )
        & df.text.str.endswith('s')
    ]
    r = df_z.sample(frac=1, random_state=0)[:100].iloc[-4]
    s, fs = read_audio_file(r.path, fs=16000)
    # res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=r.phonetics,target_word_idx=0,target_syllable_idx=-1,target_occurence_idx=0,target_phones='Z',basis='Z_Z',position='end',mode='numpy',model=default_model_charsiu)

    # res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=r.phonetics,target_word_idx=0,target_syllable_idx=-1,target_occurence_idx=0,target_phones='Z',basis='Z_Z',position='end',mode='numpy',model=default_model_charsiu)

    formatted_phonetics = prefill_for_sentence('turned around', mode='CMU')['phonetics']
    s, fs = read_audio_file('data/audio_recordings/turnEED_around.mp3', fs=16000)
    # s,fs=read_audio_file('data/audio_recordings/turned_around.mp3', fs=16000)
    default_model.predict_phone_prob_matrix(s, fs).shape
    default_model.analyze_phonetic_content(s, formatted_phonetics)

    phonetic_content = phonetic_content_analysis(s, formatted_phonetics)
    phonetic_content[
        [
            'phones',
            'start_idx',
            'end_idx',
            'pred_phones_audio',
            'proba_means',
            'GT_proba',
            'start',
            'end',
            'n_times',
        ]
    ]

    # syllable_contrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics,
    #                         target_word_idx=1,
    #                         target_syllable_idx=0,
    #                         max_speech_rate=8, mode='numpy',
    #                         to_gibberish=cmu_to_gibberish,
    #                         model=default_model
    #                 )

    prefill_for_sentence('expected', mode='MFA_IPA')['cmu_phonetics']

    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction

    model_ipa = Wav2Vec2ForFramePrediction(ipa_alphabet)
    model_ipa.load(name='model_mailabs_pca_0.95_knn_10_cos_w_UK_US_FR_ES')

    formatted_phonetics = prefill_for_sentence('turned around', mode='MFA_IPA')[
        'cmu_phonetics'
    ]

    # s,fs=librosa.load('data/audio_recordings/turnEED_around.mp3', sr=16000)
    s, fs = read_audio_file('data/audio_recordings/turned_around.mp3', fs=16000)
    model_ipa.predict_phone_prob_matrix(s, fs).shape

    s, fs = read_audio_file('data/audio_recordings/turned_around.mp3', fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=formatted_phonetics,
        target_word_idx=0,
        target_phones='d',
        basis='ɪ_d',
        model=model_ipa,
        vowels=ipa_vowels,
        consonants=ipa_consonants,
        to_gibberish=ipa_to_gibberish,
        mode='numpy',
    )

    s, fs = read_audio_file('data/audio_recordings/turnEED_around.mp3', fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=formatted_phonetics,
        target_word_idx=0,
        target_phones='D',
        model=default_model,
        mode='numpy',
    )

    path = 'data/synth_audio/cmu_words/standard/prosody/Amy/F_UK_hate.mp3'
    formatted_phonetics = prefill_for_sentence('hate')['phonetics']
    s, fs = read_audio_file(path, fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=formatted_phonetics,
        target_word_idx=0,
        target_phones='HH',
        basis='HH',
        position="start",
        mode='numpy',
    )

    path = 'data/synth_audio/cmu_words/standard/prosody/Amy/F_UK_ate.mp3'
    formatted_phonetics = prefill_for_sentence('ate')['phonetics']
    s, fs = read_audio_file(path, fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=formatted_phonetics,
        target_word_idx=0,
        target_phones='',
        basis='HH',
        position="start",
        mode='numpy',
    )

    df = actor_recordings()
    target_phones = 'T'
    selection = df[df.target_phoneme == target_phones]
    row = selection.iloc[0]
    s, fs = read_audio_file(path, fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(
        s,
        phonetics=row.cmu_phonetics,
        target_word_idx=0,
        target_phones=target_phones,
        basis='IH0_D',
        mode='numpy',
    )
