"""_summary_: This function loads different speech datasets

    Returns:
        _type_: _description_: Dataframe of different speech datasets 
"""
import pandas as pd
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from src.text_processing import remove_stress_annots
import ast
import librosa
from src.libri_phonetization_data import libri_phonetics_data
from src.wav2vec2_utils import instances_per_frame, instances_per_phoneme
from src.text_processing import prefill_for_sentence

def df_all_frames_to_X_y(df_all_frames):
    X = df_all_frames['vector'].tolist()
    y = df_all_frames['phoneme'].tolist()
    return X, y

def build_df_all_frames(df_t, phone_type, number_of_examples=100, model_path="hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft", unstressed=True):
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
    except OSError:
        # some don't have one, take a default from facebook/wav2vec2-base-960h
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained(model_path, output_hidden_states=True)

    # Here I have to shuffle. Because if there are several languages sorted and I select only some examples, it might take only examples from one language
    df_t=df_t.sample(frac=1, random_state=1)
    df_all_frames = instances_per_frame(df_t, processor, model, number_of_examples=number_of_examples, phone_type=phone_type, unstressed=unstressed)
    return df_all_frames

def build_df_all_phoneme_instances(df_t_train, phone_type='phone', number_of_examples=None, model_path="hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft", unstressed=True):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path, output_hidden_states=True)
    df_t_train['phone_df']=df_t_train.phone_df.apply(lambda r: pd.DataFrame(r))
    df_all_instances=instances_per_phoneme(df_t_train, processor, model, number_of_examples=number_of_examples, time_per_output=0.02,  phone_type=phone_type, unstressed=unstressed)
    return df_all_instances


def load_dataset_commonvoice(lang_codes=['en'], path='/home/mambauser/code/data/CommonVoice/cv-corpus-10.0-delta-2022-07-04', split="dev", phone_set='CMU'):
    """phone_set: 'CMU' or 'MFA_IPA'
    """
    df = pd.DataFrame()
    for lang_code in lang_codes:
        df_temp = pd.read_json(path+'/{}/{}_{}.json'.format(lang_code, split, phone_set))
        df_temp_metadata = pd.read_csv(path+'/{}/{}.tsv'.format(lang_code, split), sep='\t')

        df_temp_metadata=df_temp_metadata.loc[df_temp_metadata.path.isin(df_temp.filename+'.mp3')]

        map=df_temp.set_index('filename').to_dict()['phone_df']

        df_temp_metadata['phone_df']=df_temp_metadata.apply(lambda r: map[r.path.split('.')[0]], axis=1)

        for i, row in df_temp_metadata.iterrows():
            # df_temp.at[i, "path"] = '.'+row.path.split('flowchase')[1].replace('datasets', 'data')
            if type(row.phone_df) == str:
                res = ast.literal_eval(row.phone_df)
                df_temp_metadata.at[i, "phone_df"] = res

        df_temp_metadata['wav_path']=path+'/'+lang_code+'/clips/'+df_temp_metadata.path
        df = pd.concat([df, df_temp_metadata])
    
    return df



def load_dataset_edacc(path='/home/mambauser/code/data/edacc_v1.0', split="dev", phone_set='CMU'):
    """phone_set: 'CMU' or 'MFA_IPA'
    """
    df = pd.DataFrame()
    df_temp = pd.read_json('/home/mambauser/code/data/align_edacc_en_train_mfa.json')
    df_temp_metadata = pd.read_csv('/home/mambauser/code/data/edacc_v1.0/test/text_edited_new_2',sep='[;,\\t]', engine='python',header=None, names=['path', 'sentence'])

    df_temp_metadata=df_temp_metadata.loc[df_temp_metadata.path.isin(df_temp.filename+'.wav')]

    map=df_temp.set_index('filename').to_dict()['phone_df']

    df_temp_metadata['phone_df']=df_temp_metadata.apply(lambda r: map[r.path.split('.')[0]], axis=1)

    for i, row in df_temp_metadata.iterrows():
            # df_temp.at[i, "path"] = '.'+row.path.split('flowchase')[1].replace('datasets', 'data')
            if type(row.phone_df) == str:
                res = ast.literal_eval(row.phone_df)
                df_temp_metadata.at[i, "phone_df"] = res

    df_temp_metadata['wav_path']=path+'/Data_Test/'+df_temp_metadata.path
    df = pd.concat([df, df_temp_metadata])
    
    return df


#def load_dataset_edacc(path='/home/mambauser/code/data/edacc_v1.0', split="test", phone_set='CMU'):
    """phone_set: 'CMU' or 'MFA_IPA'
    """
    df = pd.DataFrame()
    df_temp = pd.read_json('/home/mambauser/code/data/align_edacc_en_train_mfa.json')
    df_temp_metadata = pd.read_csv('/home/mambauser/code/data/edacc_v1.0/test/text_edited_new_2',sep='[;,\\t]', engine='python',header=None, names=['path', 'sentence'])
    df_temp_metadata=df_temp_metadata.loc[df_temp_metadata.path.isin(df_temp.filename+'.wav')]
    map=df_temp.set_index('filename').to_dict()['phone_df']
    df_temp_metadata['phone_df']=df_temp_metadata.apply(lambda r: map[r.path.split('.')[0]], axis=1)
    for i, row in df_temp_metadata.iterrows():
            # df_temp.at[i, "path"] = '.'+row.path.split('flowchase')[1].replace('datasets', 'data')
             if type(row.phone_df) == str:
                res = ast.literal_eval(row.phone_df)
                df_temp_metadata.at[i, "phone_df"] = res

    #df_temp_metadata['wav_path']=path+'/''/clips/'+df_temp_metadata.path
    #df_temp['wav_path']='/'.join([path,'Data_Test/'])
    df_temp_metadata['wav_path']=path+'/'+'Data_Test/'+df_temp_metadata.path
    #df_temp_metadata['wav_path'] = '/'.join([path, 'Data_Test', 'example.wav'])
    #df_temp_metadata['wav_path'] = df_temp_metadata['path'].apply(lambda filename: os.path.join(path, 'Data_Test', filename))
    df = pd.concat([df, df_temp_metadata])
    
    return df


def load_dataset_MAILABS(lang_codes, path='./data/MAILABS', phone_set='CMU'):
    """phone_set: 'CMU' or 'MFA_IPA'
    """
    df_train = pd.DataFrame()
    for lang_code in lang_codes:
        df_temp = pd.read_csv(path+'/MAILABS_shuffled_aligned-{}_{}.csv'.format(lang_code, phone_set))
        for i, row in df_temp.iterrows():
            # df_temp.at[i, "path"] = '.'+row.path.split('flowchase')[1].replace('datasets', 'data')
            if type(row.phone_df) == str:
                res = ast.literal_eval(row.phone_df)
                df_temp.at[i, "phone_df"] = res

        df_train = pd.concat([df_train, df_temp])
        
    df_train = df_train.rename(columns={"path": "wav_path"}, errors="raise")
    df_train['genre']=df_train.wav_path.str.split('/').apply(lambda r: r[-5])
    df_train['speaker']=df_train.wav_path.str.split('/').apply(lambda r: r[-4])

    return df_train


def load_libri_dataset():
    df_t_train, _=libri_phonetics_data(data_set='dev-clean')
    df_t_test, _=libri_phonetics_data(data_set='test-clean')
    df_t_train["phone_df"]=df_t_train.apply(lambda r: pd.DataFrame.from_records(r.phone_df), axis=1)
    df_t_test["phone_df"]=df_t_test.apply(lambda r: pd.DataFrame.from_records(r.phone_df), axis=1)
    return df_t_train, df_t_test

def load_libri_dataset_audio_timings(df_t_test, number_of_examples=100):
    if "phone" in df_t_test.iloc[0].phone_df.columns:  df_cmu_phones=df_t_test.apply(lambda r: pd.DataFrame.from_records(r.phone_df).phone.tolist(), axis=1)
    else:  df_cmu_phones=df_t_test.apply(lambda r: pd.DataFrame.from_records(r.phone_df).cmu_phone.tolist(), axis=1)
    test_examples=[]
    for N in range(number_of_examples):
        example=df_t_test.iloc[N]
        path=example.wav_path
        example['cmu_phones']=df_cmu_phones.iloc[N]
        example['cmu_phones']=remove_stress_annots(example.cmu_phones)
        example['s'], example['fs']=librosa.load(path, sr=16000)
        test_examples.append(example)
    return pd.DataFrame(test_examples)

def load_test_dataset(df_t_test, number_of_examples=100):
    df_ipa_phones_test = pd.Series(df_t_test.phone_df.apply(lambda r: list(r['ipa_phone'].values())))
    test_examples = []
    for N in range(number_of_examples):
        example=df_t_test.iloc[N]
        path=example.wav_path
        example['ipa_phones']=df_ipa_phones_test.iloc[N]
        example['s'], example['fs']=librosa.load(path, sr=16000)
        test_examples.append(example)
    return pd.DataFrame(test_examples)

def build_df_segmented(df_t, model, forced_aligner, mode): # mode = "CMU" or "MFA_IPA"
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
        phone_prob_matrix_nonsil, _, _ = forced_aligner.get_phone_prob_matrix_nonsil(phone_prob_matrix)
        aligned_phones = forced_aligner.get_forced_alignment(phone_prob_matrix_nonsil, target_phonemes)
        pred_phones_audio, probs_means = forced_aligner.predict(aligned_phones, phone_prob_matrix_nonsil, target_phonemes)

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
        phone_prob_matrix_nonsil, _, _ = forced_aligner.get_phone_prob_matrix_nonsil(phone_prob_matrix)
        aligned_phones = forced_aligner.get_forced_alignment(phone_prob_matrix_nonsil, target_phonemes)
        predicted_phones, probs_means = forced_aligner.predict(aligned_phones, phone_prob_matrix_nonsil)
        predictions.append(predicted_phones)

    df_segmented.predicted_phones = predictions

    return df_segmented

def use_tests():
    
    df_t_train = load_dataset_MAILABS(['en_US', 'en_UK'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS')
    df_t_train=df_t_train.dropna()

    df_all_frames = build_df_all_frames(df_t_train, 'phone')
    X, y = df_all_frames_to_X_y(df_all_frames)

    df_t=load_dataset_commonvoice(lang_codes=['en'], path='./data/cv-corpus-10.0-delta-2022-07-04', split="dev", phone_set='CMU')
    df_t.accents.unique()

    accents=['United States English', 'England English,United States English', 'England English', 'Canadian English', 'England English,Canadian English']
    df_t=df_t[df_t.accents.isin(accents)]
    df_all_frames = build_df_all_frames(df_t.sample(frac=1, random_state=0), 'phone', number_of_examples=1000)
    df_all_frames.to_pickle('df_all_frames_commonvoice_en_dev_uk_us_ca_n_1000.pkl')

    accents=['England English']
    df_t=df_t[df_t.accents.isin(accents)]
    df_all_frames = build_df_all_frames(df_t.sample(frac=1, random_state=0), 'phone')
    df_all_frames.to_pickle('df_all_frames_commonvoice_en_dev_uk.pkl')

    
    df_t_test=load_dataset_commonvoice(lang_codes=['en'], path='./data/cv-corpus-10.0-delta-2022-07-04', split="test", phone_set='CMU')
    df_all_frames = build_df_all_frames(df_t_test.sample(frac=1, random_state=0), 'phone')
    df_all_frames.to_pickle('df_all_frames_commonvoice_en_test.pkl')


    