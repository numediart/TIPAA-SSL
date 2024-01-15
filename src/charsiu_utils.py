import librosa
import pandas as pd
import numpy as np

from charsiu.src.Charsiu import charsiu_forced_aligner
import torch
import numpy as np
from charsiu.src.utils import seq2duration, forced_align
from src.text_processing import (
    group_consecutive_duplicates,
    remove_stress_annots,
    phonetics_indexed_df_from_formatted_phonetics,
    unstress,
    drop_consecutive_duplicate_elements,
    drop_consecutive_duplicates,
)
from src.pronunciation_dictionaries import cmu_vowels, cmu_consonants, cmu_alphabet
from src.audio_processing import getIntonation, getIntensity, normalize

from src.dtw_forced_aligner import dtw_forced_aligner

from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction

# https://stackoverflow.com/questions/51269456/pandas-delete-consecutive-duplicates-but-keep-the-first-and-last-value
keep_first_last = lambda s: s[~((s == s.shift(1)) & (s == s.shift(-1)))]

# processing functions of df_segmented, which is the output of prediction and forced alignment


def extract_word(df_segmented, phonetics, target_word_idx):
    """phonetics must be a list of list of phonemes, e.g.: phonetics=[['AY1'],['EH1', 'N', 'D', 'IH0', 'D']]"""
    # merge lists
    phones = sum(phonetics, [])

    df_segmented = df_segmented[df_segmented.phones != '[SIL]']
    # if phones contains twice the same phone, e.g. "PhiliP Paints well", both P will be collapsed when computing the timings from DTW.
    # this results in a df_segmented shorter than "phones" list. I thus have to duplicate the corresponding row when it happens
    if len(phones) > len(df_segmented):
        # here make sure the index is a range. I will insert using .loc at i+0.5, then reset index every time
        # https://stackoverflow.com/questions/15888648/is-it-possible-to-insert-a-row-at-an-arbitrary-position-in-a-dataframe-using-pan?rq=1
        df_segmented = df_segmented.reset_index(drop=True)
        for i in range(len(phones) - 1):
            if phones[i] == phones[i + 1]:
                df_segmented.loc[i + 0.5] = df_segmented.loc[i]
                df_segmented = df_segmented.reset_index(drop=True)

    start_idx = sum([len(p) for p in phonetics][:target_word_idx])
    end_idx = sum([len(p) for p in phonetics][: target_word_idx + 1])
    df_word = df_segmented[start_idx:end_idx]
    return df_word


# get the blocks of consecutive identical rows in cols
get_blocks = lambda a, cols: a.loc[
    (a[cols].shift() == a[cols]).any(axis=1) | (a[cols].shift(-1) == a[cols]).any(axis=1)
]


class charsiu_phone_forced_aligner(charsiu_forced_aligner):
    def __init__(self, aligner, sil_threshold=4, **kwargs):
        super().__init__(aligner, sil_threshold, **kwargs)
        # this is a state variable that impact the worflow. I check for success after calling align_phone, if it's not, I return None an this status variable will also be checked in DL_speech_tech
        self.status = "success"
        self.phonetic_content = None
        self.pred_phones_audio = []
        self.fs = 16000
        self.time_per_output = 0.01

        self.p_to_id = self.charsiu_processor.processor.tokenizer.encoder
        self.id_to_p = self.charsiu_processor.processor.tokenizer.decoder

        self.forced_aligner = dtw_forced_aligner(cmu_alphabet)
        self.forced_aligner.p_to_id = self.p_to_id
        self.forced_aligner.id_to_p = self.id_to_p

    def predict_prob_matrix_and_phones(self, audio):
        audio = self.charsiu_processor.audio_preprocess(audio, sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.aligner(audio)
        cost = torch.softmax(out.logits, dim=-1).detach().cpu().numpy().squeeze()

        pred_ids_audio = torch.argmax(out.logits.squeeze(), dim=-1)
        pred_ids_audio = pred_ids_audio.detach().cpu().numpy()
        pred_phones_audio = [
            self.charsiu_processor.mapping_id2phone(int(i)) for i in pred_ids_audio
        ]

        return cost, pred_phones_audio

    def predict_phone_prob_matrix(self, s, fs):
        # this exist just for compatibility with the new pipeline and so that DL_speech_tech can call this
        cost, _ = self.predict_prob_matrix_and_phones(s)
        return cost


# To keep a compatibility with charsiu, in order to keep the possibility to quickly make comparisons, add these methods
charsiu_phone_forced_aligner.audio_to_phone_prob_matrix = (
    Wav2Vec2ForFramePrediction.audio_to_phone_prob_matrix
)
(
    charsiu_phone_forced_aligner.phone_prob_matrix_segmentation,
    _,
) = Wav2Vec2ForFramePrediction.phone_prob_matrix_segmentation


def use_tests():
    # from src.charsiu_utils import *
    from transformers import Wav2Vec2Processor
    from src.libri_phonetization_data import libri_phonetics_data
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    # from src.layer_extraction import get_last_hidden_state, get_logits, inference
    # from scripts.wav2vec2_utils import instances_per_phoneme, plot_reduction, phone_average_vectors

    # initialize model
    charsiu = charsiu_phone_forced_aligner(
        aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu'
    )
    df_t, df = libri_phonetics_data(data_set='dev-clean')
    df_phones = df_t.apply(
        lambda r: pd.DataFrame.from_records(r.phone_df).cmu_phone.tolist(), axis=1
    )

    # actor recordings
    df = pd.read_csv('data/exercise_data_export.csv')
    df.loc[:, 'audio_file_url'] = 'data/scaleway-audio-files/' + df['audio_file_url']
    # those who don't have NaN in target
    df_pContrast = df.loc[df.target_phoneme.dropna().index]
    target_phones = "T"
    selection = df_pContrast[df_pContrast.target_phoneme == target_phones]
    example = selection.iloc[10]
    # example=selection.loc[1131]
    # "ended"
    example = df.iloc[968]
    phonetics = example.cmu_phonetics

    # example=df.iloc[1539]
    # phonetics='AY1 K_AE1_N_T W_EY1|T_IH0_D F_AO1_R AW1_R S_AH1|M_ER0 R_OW1_D|T_R_IH2_P'

    path = example.audio_file_url
    s, fs = librosa.load(path, sr=16000)
    split_phonetics = sum(
        [p.replace('|', '_').split('_') for p in phonetics.split(' ')], []
    )
    # phones=[[p] for p in  remove_stress_annots(split_phonetics)]
    _, df_segmented, _ = charsiu.align_phones(audio=s, phones=split_phonetics)
    df_segmented[df_segmented.phones != '[SIL]'].GT_proba.median()
    df_segmented[df_segmented.phones != '[SIL]'].GT_proba.mean()

    df_word = charsiu.predict_word(s, [split_phonetics], 0)
    p_idx_global = -1
    GT_proba_threshold = 0.2
    phoneme_set = cmu_consonants

    phoneme_set = [[p] for p in remove_stress_annots(phoneme_set)]
    proba_means = df_word.iloc[p_idx_global].proba_means

    phoneme_set_ids = charsiu.charsiu_processor.get_phone_ids(phoneme_set)[1:-1]
    # if GT_proba is beyond the threshold, we take it as prediction
    if df_word.iloc[p_idx_global].GT_proba > GT_proba_threshold:
        phonetic_detection = target_phones
    else:
        # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
        filtered_proba_means = [
            0 if i not in phoneme_set_ids else el for i, el in enumerate(proba_means)
        ]
        idx_mean_max = np.argmax(filtered_proba_means)
        phonetic_detection = charsiu.charsiu_processor.mapping_id2phone(int(idx_mean_max))

    phonetic_content.GT_proba_means.median()
    phonetic_content.GT_proba_means.mean()

    df_segmented.apply(lambda r: r.proba_means.sum(), axis=1)

    # charsiu.predict_phone(audio=s,phones=split_phonetics)
    charsiu.predict_phone(
        s, phonetics, 0, 1, 'D', target_occurence_idx=0, phoneme_set=cmu_vowels
    )
    charsiu.predict_phone(
        s, phonetics, 0, 1, 'D', target_occurence_idx=0, phoneme_set=cmu_consonants
    )
    charsiu.predict_phone(
        s, phonetics, 0, 1, 'D', target_occurence_idx=1, phoneme_set=cmu_consonants
    )

    # df_segmented=charsiu.force_and_predict(s,split_phonetics)
    charsiu.predict_phone(s, phonetics, 0, 1, 'D')

    #  Inference with the example number N
    N = 0
    example = df.iloc[N]
    path = example.wav_path
    print(example.phones)
    example['phones'] = df_phones.iloc[N]
    # Loas an audio file
    s, fs = librosa.load(path, sr=16000)

    alignment = charsiu.align(audio=path, text=example.text)
    print(alignment)
    # _, df_segmented, _ = charsiu.align_phones(audio=s,phones=phones)
    # df_segmented=charsiu.force_and_predict(s,example.phones)
    df_segmented = df_segmented[df_segmented.phones != '[SIL]']

    # charsiu.compute_stress_score

    df_segmented[df_segmented.phones != df_segmented.pred_phones_audio]

    # ws=compute_stress_score(df_segmented, s, fs)

    charsiu.compute_stress_score(s, example.phones)

    path_to_json = '/data/audio-with-analysis-ids/data.json'
    d = pd.read_json(path_to_json)
