from src.pronunciation_dictionaries import cmu_alphabet, cmu_stressed_alphabet
import os
import numpy as np

from src.audio_processing import read_audio_file
from src.text_processing import remove_stress_annots


def test_wav2vec2_frame_prediction(n=10):
    from src.label_data_processing import actor_recordings

    df = actor_recordings()

    df_sample = df.sample(n, random_state=0)

    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction

    # --------------- Inference demo --------------------
    model = Wav2Vec2ForFramePrediction(cmu_alphabet, w2v2_model_format="onnx")
    model.load(name='model_mailabs_equilibrated_pca_95_knn_10_w')

    model_stressed = Wav2Vec2ForFramePrediction(
        cmu_stressed_alphabet, w2v2_model_format="onnx"
    )
    model_stressed.load(name='model_mailabs_equilibrated_stressed_pca_95_knn_10_cos_w')

    row = df_sample.iloc[6]

    for i, row in df_sample.iterrows():
        s, fs = read_audio_file(row.audio_file_url, fs=16000)
        # split_phonetics=sum([p.replace('|','_').split('_') for p in row.cmu_phonetics.split(' ')], [])
        split_phonetics = [
            p.replace('|', '_').split('_') for p in row.cmu_phonetics.split(' ')
        ]
        split_phonetics = sum(split_phonetics, [])

        phone_prob_matrix = model.predict_phone_prob_matrix(s, 16000)
        df_segmented = model.phone_prob_matrix_segmentation(
            phone_prob_matrix, remove_stress_annots(split_phonetics)
        )
        (
            max_posterior_df,
            max_posterior_df_filtered_processed,
            max_posterior_df_filtered_processed_threshed,
        ) = model.max_posterior_phone_df(phone_prob_matrix, proba_thresh=0.7)
        # df_segmented = model.predict_with_timings(s, remove_stress_annots(split_phonetics))
        print('model without stress df_segmented', df_segmented)
        print('model without stress max_posterior_df', max_posterior_df)
        print(
            'model without stress max_posterior_df_filtered_processed_threshed',
            max_posterior_df_filtered_processed_threshed,
        )

        phone_prob_matrix = model_stressed.predict_phone_prob_matrix(s, 16000)
        # _,_,phone_prob_df = model_stressed.audio_to_phone_prob_df(s,row.cmu_phonetics)
        df_segmented = model_stressed.phone_prob_matrix_segmentation(
            phone_prob_matrix, split_phonetics
        )
        (
            max_posterior_df,
            max_posterior_df_filtered_processed,
            max_posterior_df_filtered_processed_threshed,
        ) = model_stressed.max_posterior_phone_df(phone_prob_matrix, proba_thresh=0.7)
        # df_segmented = model_stressed.predict_with_timings(s, split_phonetics)
        print('model with stress', df_segmented)
        print(
            'model without stress max_posterior_df_filtered_processed_threshed',
            max_posterior_df_filtered_processed_threshed,
        )


def test_mfa_align(
    n=10,
    mfa_model='english_us_arpa',
    mfa_path=os.path.expanduser('~') + '/mfa_data',
    mfa_result_path=os.path.expanduser('~') + '/mfa_result',
    fs=16000,
    out_json='data/align_test.json',
):
    from src.label_data_processing import actor_recordings
    from scripts.mfa_utils import prepare_files, launch_mfa
    import shutil
    import ast
    import pandas as pd

    df = actor_recordings()
    df_sample = df.sample(n, random_state=0)

    if os.path.exists(mfa_path):
        shutil.rmtree(mfa_path)
    if os.path.exists(mfa_result_path):
        shutil.rmtree(mfa_result_path)

    prepare_files(
        "./",
        wav_paths=df_sample.audio_file_url.tolist(),
        texts=df_sample.text.tolist(),
        mfa_path=mfa_path,
        fs=fs,
    )
    textgrids_df, failures = launch_mfa(
        mfa_path, mfa_result_path, dictionary=mfa_model, acoustic_model=mfa_model
    )

    import json

    data = json.load(open(out_json))
    # as we don't know the order with which MFA will process the files, I build a dictionary from the filenames
    # for which entried will be the segmented dataframe as a dictionary, to be jsonable
    d = {}
    for f in textgrids_df.filename:
        # using "to_json()" and converting to dict with ast is necessary because python accept keys as any type where JS does not
        # so the comparison below would not work
        d[f] = ast.literal_eval(
            textgrids_df[textgrids_df.filename == f].phone_df.iloc[0].to_json()
        )

    # This tests that the predictions of the forced aligner are exactly the same for a set a 10 samples. If the acoustic model change,
    # that could actually change but would not necessarily mean that it does not work anymore but rather that the predictions are actually better
    # therefore, if that happens in the future, just update this test by running the following line to save the new predictions supposedly better
    #   json.dump(d, open(out_json, "w"))
    assert data.keys() == d.keys(), "Files list is not what it is supposed to be"

    # I think the output of timings is not purely deterministic, because running this several times lead to slightly different
    # However, checking that the sequence of phonemes at least is the same checks that mfa ran and used the dictionary the same way on texts
    for k in data.keys():
        print(d[k])
        print(data[k])
        assert (
            d[k]['phone'] == data[k]['phone']
        ), "TextGrid generated not the same as was expected. Check if it's just a change of acoustic model, or an error"

    shutil.rmtree(mfa_path)
    shutil.rmtree(mfa_result_path)
