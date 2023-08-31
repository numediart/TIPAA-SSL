from flask_server import app
import ast
from tqdm import tqdm
import pandas as pd
import requests
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.label_data_processing import actor_recordings
from DL_accuracy_performance import count_values, plot_confusion_results

from src.label_data_processing import build_user_data_df

from src.audio_processing import audio64_from_file
from src.text_processing import chunk_text, prefill_for_sentence

import base64

def call_prefill_for_sentence(sentence, mode="CMU", lang="en_US", base_url = 'http://localhost:8000', client=app.test_client()):
    url=base_url+"/prefill_from_phrase"
    data={"phrase":sentence, "mode":mode, "lang":lang}
    if client==requests: res = client.post(url,  json = json.dumps(data))
    else: res = client.post(url, 
                data=json.dumps(data),
                content_type='application/json')
    if client==requests: res.data=res._content
    d=ast.literal_eval(res.data.decode('utf8'))
    return d

# I removed this endpoint
def a_test_prefill(base_url = 'http://localhost:8000', client=app.test_client()):
    sentence="I paid a $3000 bill when visiting UCLA, it's an expensive hotel, for the 21st century!"
    d=call_prefill_for_sentence(sentence, base_url = base_url, client=client)
    
    assert d['cmu_phonetics']=='AY1 P_EY1_D AH0 {TH_R_IY1 TH_AW1|Z_AH0_N_D D_AA1|L_ER0_Z} B_IH1_L W_EH1_N V_IH1|Z_IH0|T_IH0_NG {Y_UW1 S_IY1 EH1_L EY1} IH1_T_S AE1_N IH0_K_S|P_EH1_N|S_IH0_V HH_OW0|T_EH1_L F_AO1_R DH_AH0 T_W_EH1_N|T_IY0-F_ER1_S_T S_EH1_N|CH_ER0|IY0'

    sentence="A las 22 en punto, tengo una *reunión* con el CEO, Indya, y un ingeniero de una start-up de 30000 dólares en etapa inicial, ¡luego con el CTO!"
    d=call_prefill_for_sentence(sentence, mode="MFA_IPA", lang="es_ES", base_url = base_url, client=client)
    # d=ast.literal_eval(r.data.decode('utf8'))
    sentence="At 22 o'clock, I have a *meeting* with the CEO, Indya, and an engineer of a 300 k dollars early-stage start-up, then with the CTO!"
    d=call_prefill_for_sentence(sentence, mode="MFA_IPA", lang="en_GB", base_url = base_url, client=client)
    # d=ast.literal_eval(r.data.decode('utf8'))
    sentence="A 22 heures, j'ai rendez-vous avec le CEO, Indya, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"
    d=call_prefill_for_sentence(sentence, mode="MFA_IPA", lang="fr_FR", base_url = base_url, client=client)
    # d=ast.literal_eval(r.data.decode('utf8'))

    sentence="A 22 heures, j'ai rendez-vous avec le CEO, Indya, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"
    d=call_prefill_for_sentence(sentence, mode="MFA_IPA", lang="es_ES", base_url = base_url, client=client)
    # d=ast.literal_eval(r.data.decode('utf8'))

    return d

# 'data/marie_program_all_phrases.csv'
def prefill_content_phrases(path="data/ST_content_all_phrases_08_02_2023_12_07_13.csv", path_db_export="data/query_results-2022-10-26_75652.csv"):
    all_sentences=pd.read_csv(path)
    db_export=pd.read_csv(path_db_export)

    db_phrases=set(db_export.text)
    db_phrases_lower=set(db_export.text.str.lower())
    db_phrases.union(db_phrases_lower)

    all_sentences_no_duplicates=all_sentences[~all_sentences.sentence.isin(db_phrases.union(db_phrases_lower))]
    all_sentences_no_duplicates=all_sentences_no_duplicates.drop_duplicates(['sentence'])

    no_duplicates_path=path.split('.')[0]+'_no_duplicates.csv'
    all_sentences_no_duplicates.to_csv(no_duplicates_path)


    # call_prefill_for_sentence()
    print('n of sentences:', len(all_sentences))
    # base_url="http://135.125.247.39/"
    records=[]
    empty_sentences=[]
    for i,r in tqdm(all_sentences.iterrows()):
        if r.sentence!='' and r.sentence!=' ':
            # records.append(call_prefill_for_sentence(r.sentence.strip()))
            records.append(prefill_for_sentence(r.sentence.strip()))
        else:
            empty_sentences.append(r.id)
            records.append('')
    
    idxs_errors=[i for i,el in enumerate(records) if "DOCTYPE" in el]
    # look at errors
    ids_errors=all_sentences.iloc[idxs_errors].id.tolist()
    print('List of errors:')
    print(all_sentences.iloc[idxs_errors])

    phonetics_data=pd.DataFrame.from_records(records)
    # phonetics_data['text']=all_sentences['sentence']
    # look at error rows
    print(phonetics_data[phonetics_data.index.isin(ids_errors)])
    # drop error rows
    phonetics_data=phonetics_data[~phonetics_data.index.isin(ids_errors)]

    syl_errors_df=phonetics_data[phonetics_data.apply(lambda r: len(r.n_syl_mismatches)>0, axis=1)]
    syl_errors_text=syl_errors_df.apply(lambda r: [r.segmented_text.split(' ')[i] for i in r.n_syl_mismatches], axis=1)
    syl_errors_phonetics=syl_errors_df.apply(lambda r: [r.pronounciation_guide_hr.split(' ')[i] for i in r.n_syl_mismatches], axis=1)
    syl_errors_methods=syl_errors_df.apply(lambda r: [r.used_method_for_syl_text[i] for i in r.n_syl_mismatches], axis=1)

    if len(syl_errors_df)>0: pd.DataFrame([syl_errors_text, syl_errors_phonetics, syl_errors_methods]).T

    print('List of inconsistencies in stress:')
    phonetics_data[phonetics_data.apply(lambda r: len(r.n_stress_inconsistencies)>0, axis=1)].apply(lambda r: [r.phonetics.split(' ')[i] for i in r.n_stress_inconsistencies], axis=1)

    prefill_path=path.split('.')[0]+'_prefill.csv'
    phonetics_data.to_csv(prefill_path)

    return phonetics_data


def request_for_audio_file(r, base_url = 'http://localhost:8000', endpoint='/phonemeContrast', client=requests):
    """makes a request with metadata contained in "r" and makes the call to the endpoint. It works locally or with a server, and with
    either requests module or flask's app.test_client()

    Args:
        r (dict or row of dataframe): attributes are "audio_file_url", "cmu_phonetics", "target_word_indexes", 
        "target_syllable_indexes", "alternative_phonemes", "target_phoneme"

        base_url (str, optional): [description]. Defaults to 'http://localhost:8000'.
        endpoint (str, optional): [description]. Defaults to '/phonemeContrast'.
        client ([type], optional): [description]. Defaults to requests.

    Returns:
        dict: output of the request 
    """
    url=base_url+endpoint
    print(r['audio_file_url'])

    # so that it works with Stress and stress
    # encode_string = base64.b64encode(open(r['audio_file_url'], "rb").read())
    encode_string=audio64_from_file(r['audio_file_url'])
    if 'tress' in endpoint:
        n_words_by_chunk=chunk_text(r['text'])
        p=r['cmu_phonetics']

        cumsum=0
        chunks_p=[]
        for n in n_words_by_chunk:
            chunks_p.append(' '.join(p.split(' ')[cumsum:cumsum+n])); cumsum+=n

        print(chunks_p)

        # data={"phonetics":json.dumps(chunks_p), 'audio64':encode_string.decode('utf-8')}
        data={"phonetics":chunks_p, 'audio64':encode_string.decode('utf-8')}
        # post works as well with both clients
        if client==requests: res = client.post(url,  json = json.dumps(data))
        else: res = client.post(url, 
                    data=json.dumps(data),
                    content_type='application/json')
        # if client==requests: res = client.post(url,  params = data)
        # else: res = client.post(url,  query_string = data)
    else:
        data={"phonetics":r['cmu_phonetics'], 'audio64':encode_string.decode('utf-8'), 
                                    'word_idx':int(ast.literal_eval(r['target_word_indexes'])[0]), 
                                    'syl_idx':int(ast.literal_eval(r['target_syllable_indexes'])[0]), 
                                    'target':r['target_phoneme']}
        # post works as well with both clients
        if client==requests: res = client.post(url,  json = json.dumps(data))
        else: res = client.post(url, 
                    data=json.dumps(data),
                    content_type='application/json')
        # if client==requests: res = client.post(url,  params = data)
        # else: res = client.post(url,  query_string = data)

    if client==requests: res.data=res._content

    print(res)
    print(res.data)
    assert res.status_code==200
    
    return res


def get_results(df, base_url = 'http://localhost:8000', endpoint='/v2/w2v/contrast/vowel', client=app.test_client()):
    df.index=range(len(df))
    results_records=[]
    failures=[]
    # r=df_pContrast.iloc[4]
    print(len(df))
    for i,r in tqdm(df.iterrows()):
        res=request_for_audio_file(r, base_url = base_url, endpoint=endpoint, client=client)
        assert res.status_code==200
        if 'success' in res.data.decode('utf-8'):
            # d=ast.literal_eval(res.data.decode('utf-8'))
            d=json.loads(res.data.decode('utf-8'))
            d['cmu_phonetics']=r.cmu_phonetics
            d['audio_file_url']=r.audio_file_url
            d['text']=r.text
            d['target_phoneme']=r.target_phoneme
            results_records.append(d)
        else:
            failures.append(res)
    results_df=pd.DataFrame.from_records(results_records)
    return results_df, failures

def test_actor_recordings(base_url = 'http://localhost:8000', route='/v2/w2v/contrast/', client=app.test_client(), n_ex_by_module=100):
    df=actor_recordings()

    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    vowels=['IH','IY','OW','AO','AA']
    eds=['T','D','IH0_D']

    df_v=df_pContrast[~df_pContrast.target_phoneme.isin(eds)].sample(frac=1,random_state=0).drop_duplicates(['text'])
    df_ed=df_pContrast[df_pContrast.target_phoneme.isin(eds)].sample(frac=1,random_state=0).drop_duplicates(['text'])

    
    results_v, failures_v=get_results(df_v.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'vowel', client=client)
    # results_ed_s, failures_ed_s=get_results(df_ed.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'syllable', client=client)
    results_ed, failures_ed=get_results(df_ed.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'termination', client=client)

    assert failures_v==[]
    assert failures_ed==[]

    results_v[results_v.phonetic_detection!=results_v.target_phoneme]
    
    results_ed[results_ed.phonetic_detection!=results_ed.target_phoneme]

    success_rate = lambda df : len(df[df.phonetic_detection==df.target_phoneme])/len(df) if len(df)>0 else float('nan')
    success_rate(results_ed)
    success_rate(results_v)

    for v in vowels:
        res=results_v[results_v.target_phoneme.str.contains(v)]
        print(count_values(res[res.phonetic_detection!=res.target_phoneme].phonetic_detection))
        print('success rate for '+v, success_rate(res))
    for t in eds:
        res=results_ed[results_ed.target_phoneme.str.contains(t)]
        print(count_values(res[res.phonetic_detection!=res.target_phoneme].phonetic_detection))
        print('success rate for '+t, success_rate(res))
    # for t in eds:
    #     res=results_ed_s[results_ed_s.target_phoneme.str.contains(t)]
    #     print(count_values(res[res.gibberish_detected!=res.gibberish_truth].gibberish_detected))
    #     success_rate = lambda df : len(df[df.gibberish_truth==df.gibberish_detected])/len(df) if len(df)>0 else float('nan')
    #     print('success rate for '+t, success_rate(res))
    
    # return results_v, results_ed#, results_ed_s

def test_stress_detection(base_url = 'http://localhost:8000', route='/v2/w2v/stress/', client=app.test_client(), n_ex_by_module=10):
    df=actor_recordings()
    len_t_seg=df.apply(lambda r: len(r.syllable_parts.split(' ')), axis=1)
    len_p=df.apply(lambda r: len(r.cmu_phonetics.split(' ')), axis=1)
    df[len_t_seg!=len_p]

    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    results_ss, failures_ss=get_results(df_sentence_stress.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'sentence', client=client)
    results_ws, failures_ws=get_results(df_word_stress.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'word', client=client)

    df_test=df_word_stress[df_word_stress.text.str.contains("grandma")].iloc[-1:,:]
    results_ws, failures_ws=get_results(df_test, base_url = base_url, endpoint=route+'word', client=client)

    request_for_audio_file(df_test.iloc[0], base_url = base_url, endpoint=route+'word', client=client)

    assert failures_ss==[]
    assert failures_ws==[]

    return results_ss, failures_ss, results_ws, failures_ws

def test_audio64(path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a', base_url = 'http://localhost:8000', client=app.test_client()):
    
    from src.text_processing import prefill_for_sentence

    text="I would love to go to ireland"
    r={}
    r['cmu_phonetics']=prefill_for_sentence(text)['phonetics']
    r['text']=text
    r['audio_file_url']=path
    r['target_phoneme']=float('nan')
    res=request_for_audio_file(r, base_url=base_url, endpoint='/v2/w2v/stress/sentence', client=client)
    print(res.data)
    assert res.status_code==200

def test_empty(base_url = 'http://localhost:8000', client=app.test_client()):
    
    from src.text_processing import prefill_for_sentence

    import io
    # this is an audio with 0 sample at 44100Hz
    encode_string="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EErKnINQSkqZcxBKSSmUkkpKrYVSSkqptQIAAAocAAACbNCUWByg0JCVAEAqAIDBcSzL80RRNWXZsSTPE0XTVFXbdizL80TRNFXVti3PE0XTVFXX1XXL80TRVFXVdXXdE0XVVFXXlWXf90TRNFXVdWXZ903TdFXXlWXb9n3TNFXXdWVZtn1hdVXXlWXb1m1jWFXXdWXZtm1dOW7d1nXhF4ZhmNq67vu+LwzH8EwDAMATHACACmxYHeGkaCyw0JCVAEAGAABhDEIGIYUMQkghhZRCSCklAABgwAEAIMCEMlBoyEoAIBUAACDEWmuttdZaYqm11lprrbWGSmuttdZaa6211lprrbXWWmuttdZaa6211lprrbXWWmuttZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKRUA6FfhAOD/YMPqCCdFY4GFhqwEAMIBAABjlGIMOukkpNQw5RiEUlJJpZVGMecglJJSSq1VzklIpaXWWouxck5KSSm1FluMHYSUWmotxhhj7CCklFprMcYYYyilpRhjrDHWWkNJqbUYY4w111pSai3GWmutufeSUosxxlxr7rmX1mKsteacc849tRZjrTXn3HPwqbUYY8619957UK3FWGuuOQfhewEA3A0OABAJNs6wknRWOBpcaMhKACAkAIBAiDHGnHMOQgghREox5pxzEEIIIYRIKcaccw5CCCGEkDHmnHMQQgihlFIyxpxzDkIIJZRQSuaccxBCCKGUUkrJnHMOQgghlFJKKR10EEIIoZRSSimlcw5CCKGUUkoppYQQQiillFJKKaWUEEIIpZRSSimllBJCCKWUUkoppZRSQgihlFJKKaWkUkoIoZRSSimllFJKCSGUUkoppZRSSimhhFJKKaWUUkopJZRQSimllFJKKqUUAABw4AAAEGAEnWRUWYSNJlx4AAoNWQkAAAEAIM5abClGRjHnIIbIIMQghgopxZy1DCmDHKZMKYSUlc4xhoiTFlsLFQMAAEAQAEAgZAKBAigwkAEABwgJUgBAYYGhQ4QIEKPAwLi4tAEACEJkhkhELAaJCdVAUTEdACwuMOQDQIbGRtrFBXQZ4IIu7joQQhCCEMTiAApIwMEJNzzxhifc4ASdolIHAQAAAABwAAAPAADHBhAR0RxHh8cHSIjICElJAAAAAADgAMAHAMBhAkRENMfR4fEBEiIyQlISAAAAAAAAAAAABAQEAAAAAAACAAAABARPZ2dTAAQAAAAAAAAAAOSvZQQCAAAAIz0MRgEBAA=="
    decode_string = base64.b64decode(encode_string)
    text="I would love to go to ireland"
    r={}
    r['cmu_phonetics']=prefill_for_sentence(text)['phonetics']
    r['text']=text
    r['audio_file_url']=io.BytesIO(decode_string)
    r['target_phoneme']=float('nan')
    res=request_for_audio_file(r, base_url=base_url, endpoint='/v2/w2v/stress/sentence', client=client)
    print(res.data)
    assert res.status_code==200


def pContrast_for_user_data( target_phones='AO1', n_user=10, n_ex_by_ex_type=10):
    user_data=build_user_data_df()
    
    exercise_data=pd.read_csv('data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv')
    # user_data['module_type']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].module_type.values[0], axis=1)
    user_data['target_phoneme']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].target_phoneme.values[0], axis=1)
    # user_data['cmu_phonetics']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].cmu_phonetics.values[0], axis=1)
    # user_data['target_word_indexes']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].target_word_indexes.values[0], axis=1)

    selection=user_data[user_data.target_phoneme==target_phones]
    selection['text']=selection.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].text.values[0], axis=1)
    selection['cmu_phonetics']=selection.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].cmu_phonetics.values[0], axis=1)
    selection['target_word_indexes']=selection.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].target_word_indexes.values[0], axis=1)
    selection['target_syllable_indexes']=selection.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].target_syllable_indexes.values[0], axis=1)
    selection['uid']=selection.apply(lambda r: r.exercise_id+r.audio_file_idx, axis=1)

    selection['audio_file_url']=selection['fpath']
    selections=[]
    
    for u in selection.user_id.unique()[:n_user]:
        for ex in selection.exercise_id.unique():
            selections.append(selection[selection.user_id==u][selection.exercise_id==ex][:n_ex_by_ex_type])
    df=pd.concat(selections)

    results_df, failures=get_results(df)
    results_df, failures=get_results(df, endpoint='/v2/w2v/contrast/vowel')
    

    df['phonetic_detection']=results_df['phonetic_detection']
    df['status']=results_df['status']
    df.to_csv('performance_results/vowel_contrast_user_data_'+target_phones+'_w2v.csv') 
    
    d=count_values(results_df.phonetic_detection.tolist())
    d.columns=[target_phones]

    return df, d


def vowels_confusions_user_recordings(n_user=10, n_ex_by_ex_type=10):
    vowels=['IY1','IH1','AO1','AA1','OW1']
    results={}
    for v in vowels: 
        print("vowel:",v)
        _, rates=pContrast_for_user_data(target_phones=v, n_user=n_user, n_ex_by_ex_type=n_ex_by_ex_type)
        # pickle.dump(predictions[v], open('vowel_accuracies'+v+'.p','wb'))
        results[v]=rates
    return results


def use_tests():

    # from test_DL_api import *
    base_url = 'http://localhost:8000'; route='/v2/w2v/stress/'; client=app.test_client(); n_ex_by_module=10
    df=actor_recordings()
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)
    
    results_ss, failures_ss=get_results(df_sentence_stress.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'sentence', client=client)

    df_test=df_word_stress[df_word_stress.text.str.contains("grandma")].iloc[-1:,:]
    results_ws, failures_ws=get_results(df_test, base_url = base_url, endpoint=route+'word', client=client)

    
    # # TODO: try multipart form data to send files with payload data
    # # https://stackoverflow.com/questions/12385179/how-to-send-a-multipart-form-data-with-requests-in-python
    # from requests_toolbelt.multipart.encoder import MultipartEncoder

    # mp_encoder = MultipartEncoder(
    #     fields={
    #         'phonetics': 'bar',
    #         # plain file object, no filename or mime type produces a
    #         # Content-Disposition header with just the part name
    #         'audio64': ('temp.ogg', open('data/temp.ogg', 'wb'), 'audio/ogg'),
    #     }
    # )

    # client=requests
    # client=app.test_client()
    # r = client.post(
    #     'http://localhost:8000/w2v/contrast/consonant',
    #     data=mp_encoder,  # The MultipartEncoder is posted as data, don't use files=...!
    #     # The MultipartEncoder provides the content-type header with the boundary:
    #     # headers={'Content-Type': mp_encoder.content_type}
    # )

    # from test_DL_api import *
    test_actor_recordings(base_url = 'http://localhost:8000', client=requests, n_ex_by_module=10)
    test_stress_detection(base_url = 'http://localhost:8000', client=requests, n_ex_by_module=10)

    test_actor_recordings(base_url = 'http://35.187.176.184/', client=requests, n_ex_by_module=10)
    test_stress_detection(base_url = 'http://35.187.176.184/', client=requests, n_ex_by_module=10)
    
    test_actor_recordings(base_url = 'https://spdev.flowchase.app/', client=requests, n_ex_by_module=10)
    test_stress_detection(base_url = 'http://34.77.240.102:8000', client=requests, n_ex_by_module=10)
    
    r=test_actor_recordings()
    r=test_stress_detection()

    test_audio64(path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a', base_url = 'http://135.125.247.39:8000', client=requests)
    
    df=actor_recordings()
    len_t_seg=df.apply(lambda r: len(r.syllable_parts.split(' ')), axis=1)
    len_p=df.apply(lambda r: len(r.cmu_phonetics.split(' ')), axis=1)
    df[len_t_seg!=len_p]
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    l=df_sentence_stress.apply(lambda r: len(r.text), axis=1)
    df_sentence_stress[l>50].text.iloc[0]
    df_sentence_stress[l>50].iloc[0]

    from src.audio_processing import audio64_from_file
    df_sentence_stress[l>50].iloc[0].cmu_phonetics
    audio64=audio64_from_file(df_sentence_stress[l>50].iloc[0].audio_file_url)

    

    # from src.text_processing import chunk_text
    r=df_sentence_stress[l>50].iloc[0]

    route='/v2/w2v/stress/'
    get_results(df_sentence_stress.iloc[3:4,:], endpoint=route+'sentence', client=app.test_client())
    request_for_audio_file(df_sentence_stress.iloc[3,:], endpoint=route+'sentence', client=app.test_client())

    # from test_DL_api import *;print(pContrast_for_user_data())

    import time
    start=time.time()
    results=vowels_confusions_user_recordings()
    plot_confusion_results(results, name='vowel_contrast_user_data_w2v_w_speech_corr')
    duration=time.time()-start
    print('duration:', duration) # 1800s = 30m

    # test_api()
    # r=test_GE_linguistic_data_content()
    
    # idx=r[r.status!='success'].index[0]
    # row=df_pContrast.loc[idx]
    # request_for_audio_file(row, endpoint='/w2v/contrast/phoneme', client=app.test_client())

    import requests
    # try with local docker container
    test_stress_detection(client=requests)
    # test_GE_linguistic_data_content(client=requests)

    # test_GE_linguistic_data_content(base_url = 'http://146.59.241.79/', client=requests)

    df, d=pContrast_for_user_data()
    print(d)
    