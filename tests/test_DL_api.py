import ast
import base64
import io
import json
import warnings

import pandas as pd
from tqdm import tqdm

from flowspeech.audio_processing import audio64_from_file
from flowspeech.DL_accuracy_performance import count_values
from flowspeech.label_data_processing import actor_recordings, build_user_data_df
from flowspeech.text_processing import chunk_text, prefill_for_sentence

warnings.filterwarnings("ignore", category=UserWarning)


def request_for_audio_file(client, r, endpoint='/phonemeContrast'):
    """makes a request with metadata contained in "r" and makes the call to the endpoint

    Args:
        r (dict or row of dataframe): attributes are "audio_file_url", "cmu_phonetics", "target_word_indexes",
        "target_syllable_indexes", "alternative_phonemes", "target_phoneme"

    Returns:
        dict: output of the request
    """
    # so that it works with Stress and stress
    encode_string = audio64_from_file(r['audio_file_url'])
    if 'tress' in endpoint:
        n_words_by_chunk = chunk_text(r['text'])
        p = r['cmu_phonetics']

        cumsum = 0
        chunks_p = []
        for n in n_words_by_chunk:
            chunks_p.append(' '.join(p.split(' ')[cumsum : cumsum + n]))
            cumsum += n

        data = {"phonetics": chunks_p, 'audio64': encode_string.decode('utf-8')}
    else:
        data = {
            "phonetics": r['cmu_phonetics'],
            'audio64': encode_string.decode('utf-8'),
            'word_idx': int(ast.literal_eval(r['target_word_indexes'])[0]),
            'syl_idx': int(ast.literal_eval(r['target_syllable_indexes'])[0]),
            'target': r['target_phoneme'],
        }

    res = client.post(endpoint, json=data)

    return res


def get_results(
    df,
    client,
    endpoint='/v2/w2v/contrast/vowel',
):
    df.index = range(len(df))
    results_records = []
    failures = []
    # r=df_pContrast.iloc[4]
    print(len(df))
    for i, r in tqdm(df.iterrows()):
        res = request_for_audio_file(client, r, endpoint=endpoint)
        assert res.status_code == 200
        if 'success' in res.data.decode('utf-8'):
            # d=ast.literal_eval(res.data.decode('utf-8'))
            d = json.loads(res.data.decode('utf-8'))
            d['cmu_phonetics'] = r.cmu_phonetics
            d['audio_file_url'] = r.audio_file_url
            d['text'] = r.text
            d['target_phoneme'] = r.target_phoneme
            results_records.append(d)
        else:
            failures.append(res)
    results_df = pd.DataFrame.from_records(results_records)
    return results_df, failures


def test_actor_recordings(
    client,
    route='/v2/w2v/contrast/',
    n_ex_by_module=100,
):
    df = actor_recordings()

    df_pContrast = df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress = df.loc[df.stress_category.dropna().index]
    df_word_stress = df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    vowels = ['IH', 'IY', 'OW', 'AO', 'AA']
    eds = ['T', 'D', 'IH0_D']

    df_v = (
        df_pContrast[~df_pContrast.target_phoneme.isin(eds)]
        .sample(frac=1, random_state=0)
        .drop_duplicates(['text'])
    )
    df_ed = (
        df_pContrast[df_pContrast.target_phoneme.isin(eds)]
        .sample(frac=1, random_state=0)
        .drop_duplicates(['text'])
    )

    results_v, failures_v = get_results(
        df_v.iloc[:n_ex_by_module, :],
        client=client,
        endpoint=route + 'vowel',
    )
    # results_ed_s, failures_ed_s=get_results(df_ed.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'syllable', client=client)
    results_ed, failures_ed = get_results(
        df_ed.iloc[:n_ex_by_module, :],
        client=client,
        endpoint=route + 'termination',
    )

    assert failures_v == []
    assert failures_ed == []

    results_v[results_v.phonetic_detection != results_v.target_phoneme]

    results_ed[results_ed.phonetic_detection != results_ed.target_phoneme]

    def success_rate(df):
        return (
            len(df[df.phonetic_detection == df.target_phoneme]) / len(df)
            if len(df) > 0
            else float("nan")
        )

    success_rate(results_ed)
    success_rate(results_v)

    assert success_rate(results_ed) > 0.9
    assert success_rate(results_v) > 0.7

    success_rates_vowels = {}
    for v in vowels:
        res = results_v[results_v.target_phoneme.str.contains(v)]
        print(
            count_values(
                res[res.phonetic_detection != res.target_phoneme].phonetic_detection
            )
        )
        print('success rate for ' + v, success_rate(res))
        success_rates_vowels[v] = success_rate(res)

    success_rates_terminations = {}
    for t in eds:
        res = results_ed[results_ed.target_phoneme.str.contains(t)]
        print(
            count_values(
                res[res.phonetic_detection != res.target_phoneme].phonetic_detection
            )
        )
        print('success rate for ' + t, success_rate(res))
        success_rates_terminations[t] = success_rate(res)


def test_stress_detection(
    client,
    route='/v2/w2v/stress/',
    n_ex_by_module=10,
):
    df = actor_recordings()
    len_t_seg = df.apply(lambda r: len(r.syllable_parts.split(' ')), axis=1)
    len_p = df.apply(lambda r: len(r.cmu_phonetics.split(' ')), axis=1)
    df[len_t_seg != len_p]

    df_pContrast = df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress = df.loc[df.stress_category.dropna().index]
    df_word_stress = df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    results_ss, failures_ss = get_results(
        df_sentence_stress.iloc[:n_ex_by_module, :],
        client=client,
        endpoint=route + 'sentence',
    )
    results_ws, failures_ws = get_results(
        df_word_stress.iloc[:n_ex_by_module, :],
        client=client,
        endpoint=route + 'word',
    )

    df_test = df_word_stress[df_word_stress.text.str.contains("grandma")].iloc[-1:, :]
    results_ws, failures_ws = get_results(df_test, client=client, endpoint=route + 'word')

    res = request_for_audio_file(client, df_test.iloc[0], endpoint=route + 'word')
    assert res.status_code == 200

    assert failures_ss == []
    assert failures_ws == []

    return results_ss, failures_ss, results_ws, failures_ws


def test_audio64(
    client,
    path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a',
):
    text = "I would love to go to ireland"
    r = {}
    r['cmu_phonetics'] = prefill_for_sentence(text)['phonetics']
    r['text'] = text
    r['audio_file_url'] = path
    r['target_phoneme'] = float('nan')
    res = request_for_audio_file(client, r, endpoint='/v2/w2v/stress/sentence')
    print(res.data)
    assert res.status_code == 200


def test_empty(client):
    # this is an audio with 0 sample at 44100Hz
    encode_string = "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EErKnINQSkqZcxBKSSmUkkpKrYVSSkqptQIAAAocAAACbNCUWByg0JCVAEAqAIDBcSzL80RRNWXZsSTPE0XTVFXbdizL80TRNFXVti3PE0XTVFXX1XXL80TRVFXVdXXdE0XVVFXXlWXf90TRNFXVdWXZ903TdFXXlWXb9n3TNFXXdWVZtn1hdVXXlWXb1m1jWFXXdWXZtm1dOW7d1nXhF4ZhmNq67vu+LwzH8EwDAMATHACACmxYHeGkaCyw0JCVAEAGAABhDEIGIYUMQkghhZRCSCklAABgwAEAIMCEMlBoyEoAIBUAACDEWmuttdZaYqm11lprrbWGSmuttdZaa6211lprrbXWWmuttdZaa6211lprrbXWWmuttZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKRUA6FfhAOD/YMPqCCdFY4GFhqwEAMIBAABjlGIMOukkpNQw5RiEUlJJpZVGMecglJJSSq1VzklIpaXWWouxck5KSSm1FluMHYSUWmotxhhj7CCklFprMcYYYyilpRhjrDHWWkNJqbUYY4w111pSai3GWmutufeSUosxxlxr7rmX1mKsteacc849tRZjrTXn3HPwqbUYY8619957UK3FWGuuOQfhewEA3A0OABAJNs6wknRWOBpcaMhKACAkAIBAiDHGnHMOQgghREox5pxzEEIIIYRIKcaccw5CCCGEkDHmnHMQQgihlFIyxpxzDkIIJZRQSuaccxBCCKGUUkrJnHMOQgghlFJKKR10EEIIoZRSSimlcw5CCKGUUkoppYQQQiillFJKKaWUEEIIpZRSSimllBJCCKWUUkoppZRSQgihlFJKKaWkUkoIoZRSSimllFJKCSGUUkoppZRSSimhhFJKKaWUUkopJZRQSimllFJKKqUUAABw4AAAEGAEnWRUWYSNJlx4AAoNWQkAAAEAIM5abClGRjHnIIbIIMQghgopxZy1DCmDHKZMKYSUlc4xhoiTFlsLFQMAAEAQAEAgZAKBAigwkAEABwgJUgBAYYGhQ4QIEKPAwLi4tAEACEJkhkhELAaJCdVAUTEdACwuMOQDQIbGRtrFBXQZ4IIu7joQQhCCEMTiAApIwMEJNzzxhifc4ASdolIHAQAAAABwAAAPAADHBhAR0RxHh8cHSIjICElJAAAAAADgAMAHAMBhAkRENMfR4fEBEiIyQlISAAAAAAAAAAAABAQEAAAAAAACAAAABARPZ2dTAAQAAAAAAAAAAOSvZQQCAAAAIz0MRgEBAA=="
    decode_string = base64.b64decode(encode_string)
    text = "I would love to go to ireland"
    r = {}
    r['cmu_phonetics'] = prefill_for_sentence(text)['phonetics']
    r['text'] = text
    r['audio_file_url'] = io.BytesIO(decode_string)
    r['target_phoneme'] = float('nan')
    res = request_for_audio_file(client, r, endpoint='/v2/w2v/stress/sentence')
    print(res.data)
    assert res.status_code == 200


def pContrast_for_user_data(target_phones='AO1', n_user=10, n_ex_by_ex_type=10):
    user_data = build_user_data_df()

    exercise_data = pd.read_csv(
        'data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv'
    )
    # user_data['module_type']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].module_type.values[0], axis=1)
    user_data['target_phoneme'] = user_data.apply(
        lambda r: exercise_data[
            exercise_data.exercise_id == r.exercise_id
        ].target_phoneme.values[0],
        axis=1,
    )
    # user_data['cmu_phonetics']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].cmu_phonetics.values[0], axis=1)
    # user_data['target_word_indexes']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].target_word_indexes.values[0], axis=1)

    selection = user_data[user_data.target_phoneme == target_phones]
    selection['text'] = selection.apply(
        lambda r: exercise_data[exercise_data.exercise_id == r.exercise_id].text.values[
            0
        ],
        axis=1,
    )
    selection['cmu_phonetics'] = selection.apply(
        lambda r: exercise_data[
            exercise_data.exercise_id == r.exercise_id
        ].cmu_phonetics.values[0],
        axis=1,
    )
    selection['target_word_indexes'] = selection.apply(
        lambda r: exercise_data[
            exercise_data.exercise_id == r.exercise_id
        ].target_word_indexes.values[0],
        axis=1,
    )
    selection['target_syllable_indexes'] = selection.apply(
        lambda r: exercise_data[
            exercise_data.exercise_id == r.exercise_id
        ].target_syllable_indexes.values[0],
        axis=1,
    )
    selection['uid'] = selection.apply(lambda r: r.exercise_id + r.audio_file_idx, axis=1)

    selection['audio_file_url'] = selection['fpath']
    selections = []

    for u in selection.user_id.unique()[:n_user]:
        for ex in selection.exercise_id.unique():
            selections.append(
                selection[selection.user_id == u][selection.exercise_id == ex][
                    :n_ex_by_ex_type
                ]
            )
    df = pd.concat(selections)

    results_df, failures = get_results(df)
    results_df, failures = get_results(df, endpoint='/v2/w2v/contrast/vowel')

    df['phonetic_detection'] = results_df['phonetic_detection']
    df['status'] = results_df['status']
    df.to_csv(
        'performance_results/vowel_contrast_user_data_' + target_phones + '_w2v.csv'
    )

    d = count_values(results_df.phonetic_detection.tolist())
    d.columns = [target_phones]

    return df, d


def vowels_confusions_user_recordings(n_user=10, n_ex_by_ex_type=10):
    vowels = ['IY1', 'IH1', 'AO1', 'AA1', 'OW1']
    results = {}
    for v in vowels:
        print("vowel:", v)
        _, rates = pContrast_for_user_data(
            target_phones=v, n_user=n_user, n_ex_by_ex_type=n_ex_by_ex_type
        )
        # pickle.dump(predictions[v], open('vowel_accuracies'+v+'.p','wb'))
        results[v] = rates
    return results
