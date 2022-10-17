import pandas as pd
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2ForCTC
from utils.text_processing import remove_stress_annots
import ast
import json
import librosa
from sklearn.model_selection import train_test_split
from utils.libri_phonetization_data import libri_phonetics_data
from utils.wav2vec2_espeak import instances_per_frame
from utils.text_processing import prefill_for_sentence
from itertools import groupby

def df_all_frames_to_X_y(df_all_frames):
    X = df_all_frames['vector'].tolist()
    y = df_all_frames['phoneme'].tolist()
    return X, y

def build_df_all_frames(df_t, phone_type):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft", output_hidden_states=True)
    df_all_frames = instances_per_frame(df_t, processor, model, phone_type=phone_type)
    return df_all_frames

def load_ipa_dataset(lang_code):  
    df_t = pd.read_csv('./data/MAILABS_aligned-{}.csv'.format(lang_code))

    for i, row in df_t.iterrows():
        df_t.at[i, "wav_path"] = './data/MAILABS/{}/{}.wav'.format(lang_code, row.filename)
        if type(row.phone_df) == str:
            res = ast.literal_eval(row.phone_df)
            df_t.at[i, "phone_df"] = res

    df_t_train, df_t_test = train_test_split(df_t, test_size=0.2)

    return df_t_train, df_t_test

def leave_one_speaker_out(speaker_lang_code, others, train_size=800, test_size=200):
    df_others = pd.DataFrame()
    for lang_code in others:
        df_temp = pd.read_csv('./data/MAILABS_shuffled_aligned-{}.csv'.format(lang_code))
        for i, row in df_temp.iterrows():
            if type(row.phone_df) == str:
                res = ast.literal_eval(row.phone_df)
                df_temp.at[i, "phone_df"] = res
            if type(row.phone_df) == float:
                df_temp = df_temp.drop(i)

        df_others = pd.concat([df_others, df_temp])
    df_others = df_others.rename(columns={"path": "wav_path"}, errors="raise")

    df_speaker = pd.read_csv('./data/MAILABS_shuffled_aligned-{}.csv'.format(speaker_lang_code))
    for i, row in df_speaker.iterrows():
        if type(row.phone_df) == str:
            res = ast.literal_eval(row.phone_df)
            df_speaker.at[i, "phone_df"] = res
        if type(row.phone_df) == float:
            df_speaker = df_speaker.drop(i)

    df_speaker = df_speaker.rename(columns={"path": "wav_path"}, errors="raise")

    df_train = df_others.sample(train_size)
    df_train['wav_path'] = df_train['wav_path'].apply(lambda x : './data/MAILABS/'+x)
    df_test = df_speaker.sample(test_size)
    df_test['wav_path'] = df_test['wav_path'].apply(lambda x : './data/MAILABS/'+x)

    return df_train, df_test

def load_shuffled_ipa_dataset(lang_code):
    df_t = pd.read_csv('./data/MAILABS_shuffled_aligned-{}.csv'.format(lang_code))
    df_t.rename(columns={"path": "wav_path"}, errors="raise")
    df_t_train, df_t_test = train_test_split(df_t, test_size=0.2)
    return df_t_train, df_t_test

def load_cmu_dataset():
    df_t_train, df_train=libri_phonetics_data(data_set='dev-clean')
    df_t_test, df_test=libri_phonetics_data(data_set='test-clean')
    df_t_test.apply(lambda r: pd.DataFrame.from_records(r.phone_df).phone.tolist(), axis=1)
    return df_t_train, df_t_test

def load_cmu_dataset_MAILABS(speaker_lang_code, others, train_size=800, test_size=200):
    df_others = pd.DataFrame()
    for lang_code in others:
        df_temp = pd.read_csv('./data/MAILABS_shuffled_aligned-{}_CMU.csv'.format(lang_code))
        for i, row in df_temp.iterrows():
            df_temp.at[i, "path"] = '.'+row.path.split('flowchase')[1].replace('datasets', 'data')
            if type(row.phone_df) == str:
                res = ast.literal_eval(row.phone_df)
                df_temp.at[i, "phone_df"] = res

        df_others = pd.concat([df_others, df_temp])
    df_others = df_others.rename(columns={"path": "wav_path"}, errors="raise")

    df_speaker = pd.read_csv('./data/MAILABS_shuffled_aligned-{}_CMU.csv'.format(speaker_lang_code))
    for i, row in df_speaker.iterrows():
        df_speaker.at[i, "path"] = '.'+row.path.split('flowchase')[1].replace('datasets', 'data')
        if type(row.phone_df) == str:
            res = ast.literal_eval(row.phone_df)
            df_speaker.at[i, "phone_df"] = res
        if type(row.phone_df) == float:
            df_speaker = df_speaker.drop(i)

    df_speaker = df_speaker.rename(columns={"path": "wav_path"}, errors="raise")

    df_train = df_others.sample(train_size)
    #df_train['wav_path'] = df_train['wav_path'].apply(lambda x : './data/MAILABS/'+x)
    df_test = df_speaker.sample(test_size)
    #df_test['wav_path'] = df_test['wav_path'].apply(lambda x : './data/MAILABS/'+x)

    return df_train, df_test

def load_cmu_test_dataset(df_t_test, number_of_examples=100, random=False):
    df_cmu_phones=df_t_test.apply(lambda r: pd.DataFrame.from_records(r.phone_df).phone.tolist(), axis=1)
    test_examples=[]
    for N in range(number_of_examples):
        example=df_t_test.iloc[N]
        path=example.wav_path
        example['cmu_phones']=df_cmu_phones.iloc[N]
        example['cmu_phones']=remove_stress_annots(example.cmu_phones)
        example['s'], example['fs']=librosa.load(path, sr=16000)
        test_examples.append(example)
    return pd.DataFrame(test_examples)

def load_test_dataset(df_t_test, number_of_examples=100, random=False):
    df_ipa_phones_test = pd.Series(df_t_test.phone_df.apply(lambda r: list(r['ipa_phone'].values())))
    test_examples = []
    for N in range(number_of_examples):
        example=df_t_test.iloc[N]
        path=example.wav_path
        example['ipa_phones']=df_ipa_phones_test.iloc[N]
        example['s'], example['fs']=librosa.load(path, sr=16000)
        test_examples.append(example)
    return pd.DataFrame(test_examples)

def build_df_segmented(df_t, model, forced_aligner, mode, language_code): # mode = "CMU" or "MFA_IPA"
    df_segmented = pd.DataFrame(columns=['phones', 'pred_phones_audio', 'start', 'end'])

    for _, row in df_t.iterrows():
        df = pd.DataFrame(columns=['phones', 'pred_phones_audio', 'start', 'end'])
        if mode=='CMU':
            df.phones = remove_stress_annots(row.phone_df['cmu_phone'].values())
        else:
            df.phones = row.phone_df['ipa_phone'].values()
        
        df.start = row.phone_df['start'].values()
        df.end = row.phone_df['end'].values()

        s, fs = librosa.load(row.wav_path, sr=16000)
        target_phonemes = df.phones
        phone_prob_matrix = model.predict_phone_prob_matrix(s, fs)
        cost_nonsil, _, _ = forced_aligner.get_cost_non_sil(phone_prob_matrix)
        aligned_phones = forced_aligner.get_forced_alignment(cost_nonsil, target_phonemes)
        pred_phones_audio, probs_means = forced_aligner.predict(aligned_phones, cost_nonsil, target_phonemes)

        df.pred_phones_audio = pred_phones_audio

        df_segmented = pd.concat([df_segmented, df])

    return df_segmented

def build_df_segmented_all(df_t, model, forced_aligner, mode, language_code): # mode = "CMU" or "MFA_IPA"
    df_segmented = pd.DataFrame(columns=['phones', 'predicted_phones', 'start', 'end', 'wav_path', 'target_phonemes'])

    df_segmented.start = df_t.phone_df.apply(lambda r: list(r['start'].values()))
    df_segmented.end = df_t.phone_df.apply(lambda r: list(r['end'].values()))
    if mode=='CMU':
        df_segmented.target_phonemes=df_t.phone_df.apply(lambda r: list(r['cmu_phone'].values()))
    else:
        df_segmented.target_phonemes=df_t.phone_df.apply(lambda r: list(r['ipa_phone'].values()))
    df_segmented.wav_path = df_t.wav_path

    phones = []
    for _, r in df_t.iterrows():
        phones.append(prefill_for_sentence(sentence=r.text, lang=language_code, mode=mode)['cmu_phonetics'])
    df_segmented.phones = phones

    predictions = []
    for _, ex in df_segmented.iterrows():
        s, fs = librosa.load(ex.wav_path, sr=16000)
        target_phonemes = ex.target_phonemes
        phone_prob_matrix = model.predict_phone_prob_matrix(s, fs)
        cost_nonsil, _, _ = forced_aligner.get_cost_non_sil(phone_prob_matrix)
        aligned_phones = forced_aligner.get_forced_alignment(cost_nonsil, target_phonemes)
        predicted_phones, probs_means = forced_aligner.predict(aligned_phones, cost_nonsil)
        predictions.append(predicted_phones)

    df_segmented.predicted_phones = predictions

    return df_segmented