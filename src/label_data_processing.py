import pandas as pd
import numpy as np
import pdb
from glob import glob
from src.text_processing import (
    cmu_ensure_phonetics_consistency,
    remove_special_characters,
    remove_stress_annots,
    prefill_content,
    prefill_for_sentence,
)

from syllabipy.sonoripy import SonoriPy

from src.pronunciation_dictionaries import cmudict_dict

import itertools
import os
import ast

target_to_alternatives = {
    "DH": ["DH", "TH"],
    "TH": ["DH", "TH"],
    "AO1": ["AO1", "OW1"],
    "OW1": ["AO1", "OW1"],
    "IH1": ["IH1", "IY1"],
    "IY1": ["IH1", "IY1"],
    # "IH0 D":['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']
    "IH0 D": ['T', 'D', 'IH0 D'],
    "D": ['T', 'D', 'IH0 D'],
    "T": ['T', 'D', 'IH0 D'],
}


graphemes_to_alternatives = {"ie": ["IY1", "AY1"], "ea": ["IY1", "EH1"]}


# New database content processing
def build_data(F, n_F, df_focus_word, df_data):
    glob_F = glob(F + '/*')
    records = []
    for i, el in enumerate(n_F):
        # print(i)
        # print(el)
        id = df_focus_word.iloc[int(el) - 1]['id filipe']
        bins = df_data[df_data.id == id].bins.values[0]
        text = df_data[df_data.id == id].text.values[0]
        audio_path = glob_F[i]
        r = {
            'bins': bins,
            'text': text,
            'audio_path': audio_path,
            'n_marker': el,
            'id': id,
        }
        records.append(r)
    return pd.DataFrame.from_records(records)


def get_data_stressed_content():
    F2 = "data/new_content_audio/F2/Repetition_Tasks_Jessie_AmE/Repetition Task 1 (multivoices)"
    F1 = "data/new_content_audio/F1/Repetition Task 1"
    M1 = "data/new_content_audio/M1/Repetition Task 1"

    glob_F1 = glob(F1 + '/*')
    glob_M1 = glob(M1 + '/*')
    glob_F2 = glob(F2 + '/*')

    # marker number: odd=question, even=answer
    n_F2 = [el.split('/')[-1].split('.')[0].split('_')[-1] for el in glob_F2]
    n_M1 = [
        n if n[0] != '0' else n[1:]
        for n in [el.split('/')[-1].split('.')[0].split('_')[-1][9:] for el in glob_M1]
    ]
    n_F1 = [el.split('/')[-1].split('.')[0].split('_')[-1][6:] for el in glob_F1]

    df_data = pd.read_csv('data/syllabus_phrases_export_2021-09-15_125529.csv')
    df_data['bins'] = df_data.apply(
        lambda r: [int('*' in el) for el in r.text.split(' ')], axis=1
    )
    df2 = pd.read_csv('data/focus_word_phrases_id.csv')

    df_focus_word = df2[df2.name.str.startswith('Task1')]
    df_focus_word.index += 1

    df_F1 = build_data(F1, n_F1, df_focus_word, df_data)
    df_F2 = build_data(F2, n_F2, df_focus_word, df_data)
    df_M1 = build_data(M1, n_M1, df_focus_word, df_data)

    df_tot = pd.concat([df_F1, df_F2, df_M1], axis=0)

    return df_tot


# Actor recordings 31/03 (GE and BE)
def actor_recordings():
    df = pd.read_csv('data/flwc-phrase-audios/flwc-phrase-audios.csv')
    df['audio_file_url'] = 'data/flwc-phrase-audios/' + df['audio_file_url']

    # those who don't have NaN in target_phoneme
    # df_pContrast=df.loc[df.target_phoneme.dropna().index]
    # df_sentence_stress=df.loc[df.stress_category.dropna().index]
    # df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    return df


# User recordings data
def build_user_data_df(path='data/user_data_df.csv'):
    if not os.path.exists(path):
        # there are files at this level (when there is no date), I ignore them for now, it's like 2% of the files
        # all_files=glob('data/flwc-recordings/*/*')

        all_files = glob('data/flwc-recordings/*/*/*')
        len(all_files)

        # different possible extensions in dataset
        extensions = set([f.split('.')[-1] for f in all_files if '.' in f])

        # get filenames
        fpaths = [f for f in all_files if f.split('.')[-1] in extensions]
        user_ids = [f.split('/')[-3] for f in fpaths]
        # user_ids=set([f.split('/')[-2] for f in fpaths])

        fnames = [
            os.path.split(f.split('.')[0])[-1]
            for f in all_files
            if f.split('.')[-1] in extensions
        ]
        len(fpaths)
        len(fnames)

        # big number corresponding to the specific audio file
        audio_file_idx = [f.split('.')[0].split('__')[-1] for f in fnames]

        # remove user info from filename because it can contain '__' which is the separator and thus messes with parsing if not removed
        # and I keep the information inside user_ids list
        # metadata=[f.replace(user_ids[i]+'__', '') for i,f in enumerate(fnames)]
        # len(metadata)

        # split with '__' separator of metadata, discard last columns because user_id may contain '__' which messes the rest of the columns
        df = pd.DataFrame([f.split('__') for f in fnames]).iloc[:, :3]
        df.columns = ['exercise_id', 'processing_status', 'answer']
        df['user_id'] = user_ids
        df['audio_file_idx'] = audio_file_idx
        df['fpath'] = fpaths
        df['fname'] = fnames

        # 0=error,    1=success
        df.loc[df.processing_status == 'P0', 'processing_status'] = 0
        df.loc[df.processing_status == 'P1', 'processing_status'] = 1

        # 0=error,    1=  , 2=
        df.loc[df.answer == 'A0', 'answer'] = 0
        df.loc[df.answer == 'A1', 'answer'] = 1
        df.loc[df.answer == 'A2', 'answer'] = 2

        # df[df.processing_status==0]
        # df[df.processing_status==1]

        # df[df.answer==0]
        # df[df.answer==1]
        # df[df.answer==2]

        # df[df.processing_status==0][df.answer==0]
        # df[df.processing_status==0][df.answer!=0]

        df.to_csv(path)
    else:
        df = pd.read_csv(path)

    exercise_data = pd.read_csv(
        'data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv'
    )

    exercise_id_to_target_phoneme = dict(
        zip(exercise_data.exercise_id, exercise_data.target_phoneme)
    )
    exercise_id_to_cmu_phonetics = dict(
        zip(exercise_data.exercise_id, exercise_data.cmu_phonetics)
    )
    exercise_id_to_target_word_indexes = dict(
        zip(exercise_data.exercise_id, exercise_data.target_word_indexes)
    )
    exercise_id_to_target_syllable_indexes = dict(
        zip(exercise_data.exercise_id, exercise_data.target_syllable_indexes)
    )

    # user_data['module_type']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].module_type.values[0], axis=1)
    df['target_phoneme'] = df.apply(
        lambda r: exercise_id_to_target_phoneme[r.exercise_id], axis=1
    )
    df['cmu_phonetics'] = df.apply(
        lambda r: exercise_id_to_cmu_phonetics[r.exercise_id], axis=1
    )
    df['target_word_indexes'] = df.apply(
        lambda r: exercise_id_to_target_word_indexes[r.exercise_id], axis=1
    )
    df['target_syllable_indexes'] = df.apply(
        lambda r: exercise_id_to_target_syllable_indexes[r.exercise_id], axis=1
    )
    df['uid'] = df.apply(lambda r: str(r.exercise_id) + str(r.audio_file_idx), axis=1)

    return df


def get_errors_examples():
    df = build_user_data_df()
    df_errors = df[df.processing_status == 0]
    len(df_errors) / len(df)

    ex_ids = df_errors.exercise_id.unique()

    n_errors = {}
    for ex_id in ex_ids:
        n_errors[ex_id] = len(df_errors[df_errors.exercise_id == ex_id])

    exercise_data = pd.read_csv(
        'data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv'
    )
    df_errors['module_type'] = df_errors.exercise_id.apply(
        lambda r: exercise_data[exercise_data.exercise_id == r].module_type.values[0]
    )

    # sort dict by value
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    exs_sort_by_n_errors = list(
        {k: v for k, v in sorted(n_errors.items(), key=lambda item: item[1])}
    )[::-1]

    return df_errors, exs_sort_by_n_errors


def final_s_artificial_data(path="data/Final s - voices for test/exercises_test.csv"):
    df = pd.read_csv(path)
    df['path'] = os.path.split(path)[0] + '/audios/' + df.soundfiles_name
    content, df_errors = prefill_content(df["Text with target"].tolist())
    df['cmu_phonetics'] = content.cmu_phonetics
    content['path'] = df.path
    content['text'] = df["Text with target"]

    content.apply(lambda r: sum(['*' in el for el in r.text.split()]), axis=1)

    content['word_idx'] = 0
    sents = content[content.text.str.contains('\*')].text.str.split(' ')
    # checks in each word if there is a "*", put one if true in a list. Then I use index() to know where is the 1
    content.loc[sents.index, 'word_idx'] = sents.apply(
        lambda r: [int('*' in el) for i, el in enumerate(r)].index(1)
    )

    # content["target"]=df["Target phoneme"]

    # content["target_syllable_indexes"]=-1
    content["target_syllable_indexes"] = [[-1]] * len(content)

    # Don't use the annotations, they are wrong, I'll use cmudict instead
    # to_cmu={"/z/":"Z", "/s/":"S", "/iz/": "IH_Z"}

    # content["target_phones"]=content["target"].apply(lambda r: to_cmu[r])

    return content


def select_accent(df, accent=None):
    if accent != None:
        if accent == 'UK':
            df = df[df.path.apply(lambda r: '_UK_' in r.split('/')[-1])]
        elif accent == "US":
            df = df[df.path.apply(lambda r: '_US_' in r.split('/')[-1])]
        else:
            raise "accent must be US or UK or None"
    return df


def synth_words_data(
    path="data/synth_audio/cmu_words/standard/prosody/",
    phonetic_dict=cmudict_dict,
    mode='CMU',
    accent=None,
):
    # path="data/synth_audio/mfa_words/standard/prosody/fr_FR"
    if not os.path.exists(path + '/linguistic_data.csv'):
        audios_path = path + "/*/*"
        paths = glob(audios_path)

        df = pd.DataFrame()
        df['path'] = paths
        df['text'] = df.apply(
            lambda r: os.path.split(r.path)[-1].split('.')[0].split('_')[-1], axis=1
        )

        from tqdm import tqdm

        tqdm.pandas()
        df['phonetics'] = df.progress_apply(
            lambda r: phonetic_dict[r.text] if r.text in phonetic_dict else float('nan'),
            axis=1,
        )

        df = df.dropna()

        # not sure why, it seems there are empty entries in cmudict
        df = df[df.apply(lambda r: len(r.phonetics), axis=1) > 0]

        df['syl_p'] = df['phonetics'].progress_apply(
            lambda r: SonoriPy(r[0], mode=mode)[0]
        )
        df['phonetics'] = df['syl_p'].apply(
            lambda p: '|'.join(['_'.join(syl) for syl in p])
        )
        # df.apply(lambda r: prefill_for_sentence(r.text), axis=1)

        df.to_csv(path + '/linguistic_data.csv')
    else:
        df = pd.read_csv(path + '/linguistic_data.csv')
        # pd.read_csv()
        df['syl_p'] = df['syl_p'].apply(ast.literal_eval)

    df = df.dropna()
    df = select_accent(df, accent=accent)
    df['phonetics'] = df.apply(
        lambda r: cmu_ensure_phonetics_consistency(r.phonetics), axis=1
    )
    # recompute syl_p, because I modified phonetics with CH and JH
    df['syl_p'] = df.phonetics.str.split('|').apply(
        lambda r: [syl.split('_') for syl in r]
    )
    df['target_word_indexes'] = 0
    df['fpath'] = df['path']

    return df
