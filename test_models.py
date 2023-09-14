from src.load_data import load_libri_dataset, df_all_frames_to_X_y, load_libri_dataset_audio_timings
from src.pronunciation_dictionaries import cmu_alphabet, ipa_alphabet, cmu_phones_info, cmu_reducer, cmu_stressed_alphabet
import pandas as pd
import numpy as np
import pickle
import os
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
# from umap.umap_ import UMAP

from src.audio_processing import prepare_audio_file, read_audio_file
from src.text_processing import prefill_for_sentence, get_augmented_mfa_dict, chunk_text, remove_stress_annots

def test_wav2vec2_frame_prediction(n=10):
    
    from src.label_data_processing import actor_recordings
    df=actor_recordings()

    df_sample=df.sample(n,random_state=0)

    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction
    # --------------- Inference demo --------------------
    model = Wav2Vec2ForFramePrediction(cmu_alphabet,w2v2_model_format="onnx")
    model.load(name='model_mailabs_equilibrated_pca_95_knn_10_w')

    
    model_stressed = Wav2Vec2ForFramePrediction(cmu_stressed_alphabet,w2v2_model_format="onnx")
    model_stressed.load(name='model_mailabs_equilibrated_stressed_pca_95_knn_10_cos_w')
    
    row=df_sample.iloc[0]

    for i,row in df_sample.iterrows():
        s,fs=read_audio_file(row.audio_file_url, fs=16000)
        split_phonetics=sum([p.replace('|','_').split('_') for p in row.cmu_phonetics.split(' ')], [])
        split_phonetics=[p.replace('|','_').split('_') for p in row.cmu_phonetics.split(' ')]
        split_phonetics=sum(split_phonetics,[])

        phone_prob_matrix = model.predict_phone_prob_matrix(s, 16000)
        df_segmented=model.phone_prob_matrix_segmentation(phone_prob_matrix, remove_stress_annots(split_phonetics))
        # df_segmented = model.predict_with_timings(s, remove_stress_annots(split_phonetics))
        print('model without stress', df_segmented)

        phone_prob_matrix = model_stressed.predict_phone_prob_matrix(s, 16000)
        # _,_,phone_prob_df = model_stressed.audio_to_phone_prob_df(s,row.cmu_phonetics)
        df_segmented=model_stressed.phone_prob_matrix_segmentation(phone_prob_matrix, split_phonetics)
        max_posterior_df, max_posterior_df_filtered_collapsed=model_stressed.max_posterior_phone_df(phone_prob_matrix, proba_thresh=0.8)
        # df_segmented = model_stressed.predict_with_timings(s, split_phonetics)
        print('model with stress', df_segmented)


# For now, it works locally but not on the github actions server because of a write permission issue
def test_mfa_align(n=10, mfa_model='english_us_arpa', mfa_path=os.path.expanduser('~')+'/mfa_data', mfa_result_path=os.path.expanduser('~')+'/mfa_result', fs=16000, out_json='data/align_test.json'):
    from src.label_data_processing import actor_recordings
    from scripts.mfa_utils import prepare_files, launch_mfa
    import shutil
    df=actor_recordings()
    df_sample=df.sample(n,random_state=0)

    if os.path.exists(mfa_path):
        shutil.rmtree(mfa_path)
    if os.path.exists(mfa_result_path):
        shutil.rmtree(mfa_result_path)
        
    prepare_files("./", wav_paths=df.audio_file_url.tolist(), texts=df_sample.text.tolist(), mfa_path=mfa_path, fs=fs)
    textgrids_df, failures=launch_mfa(mfa_path, mfa_result_path, dictionary=mfa_model, acoustic_model=mfa_model)

    import json
    data = json.load(open(out_json))

    # df['phone_df']=phone_dfs
    

    import ast
    # This tests that the predictions of the forced aligner are exactly the same for a set a 10 samples. If the acoustic model change, 
    # that could actually change but would not necessarily mean that it does not work anymore but rather that the predictions are actually better
    # therefore, if that happens in the future, just update this test by running the following line to save the new predictions supposedly better
    # textgrids_df.to_json(out_json)
    assert ast.literal_eval(textgrids_df.to_json())==data, "TextGrid generated not the same as was expected. Check if it's just a change of acoustic model, or an error"

    
    shutil.rmtree(mfa_path)
    shutil.rmtree(mfa_result_path)