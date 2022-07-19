from flask_server import app
import os
os.environ['FLOWSPEECH_KEY']="ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"
import ast
from tqdm import tqdm
import pandas as pd
import requests
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from utils.label_data_processing import actor_recordings
from DL_accuracy_performance import count_values, plot_confusion_results

from utils.label_data_processing import build_user_data_df
exercise_data=pd.read_csv('data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv')

from utils.audio_processing import audio64_from_file
from utils.text_processing import get_chunks, chunk_text

def request_for_audio_file(r, base_url = 'http://localhost:8000', endpoint='/phonemeContrast', client=requests, mode='v1'):
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
    if mode=='v1':
        
        def send_audio_base64(path='data/audio_recordings/WS_111_toothpaste.wav', base_url = 'http://localhost:8000', client=requests):
            # based on :
            # https://stackoverflow.com/questions/50279380/how-to-decode-base64-string-directly-to-binary-audio-format
            encode_string = base64.b64encode(open(path, "rb").read())
            res = client.post(base_url+"/send_base64_audio", data={"audio":encode_string, "API_KEY":"ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"})

            # This is for compatibility between requests module and flask's test_client
            if client==requests: res.data=res._content

            return res
        res=send_audio_base64(path=r['audio_file_url'], base_url = base_url, client=client)
        print(res)
        print(res.data)
        assert res.status_code == 200
        # This is for compatibility between requests module and flask's test_client
        if client==requests: res.data=res._content
        res=ast.literal_eval(res.data.decode('utf-8'))
        assert res['status']=='success'
        rID=res['rID']
        print(res)

        # so that it works with Stress and stress
        if 'tress' in endpoint:
            res = client.post(url, data={"phonetics":r['cmu_phonetics'],"text":r['text'], 'rID':rID, 
                                        'target':r['target_phoneme']})
        else:
            res = client.post(url, data={"phonetics":r['cmu_phonetics'], 'rID':rID, 
                                        'word_idx':str(ast.literal_eval(r['target_word_indexes'])[0]), 
                                        'syl_idx':str(ast.literal_eval(r['target_syllable_indexes'])[0]), 
                                        'target':r['target_phoneme']})
    else:
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

    print(res.data)
    assert res.status_code==200
    
    return res


def get_results(df, base_url = 'http://localhost:8000', endpoint='/v2/w2v/contrast/vowel', client=app.test_client(), mode='v2'):
    df.index=range(len(df))
    results_records=[]
    failures=[]
    # r=df_pContrast.iloc[4]
    print(len(df))
    for i,r in tqdm(df.iterrows()):
        res=request_for_audio_file(r, base_url = base_url, endpoint=endpoint, client=client, mode=mode)
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

def test_actor_recordings(base_url = 'http://localhost:8000', route='/v2/w2v/contrast/', client=app.test_client(), n_ex_by_module=10, mode='v2'):
    df=actor_recordings()

    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    vowels=['IH','IY','OW','AO','AA']
    eds=['T','D','IH0_D']

    df_v=df_pContrast[~df_pContrast.target_phoneme.isin(eds)]
    df_ed=df_pContrast[df_pContrast.target_phoneme.isin(eds)]

    
    results_v, failures_v=get_results(df_v.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'vowel', client=client, mode=mode)
    results_ed_s, failures_ed_s=get_results(df_ed.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'syllable', client=client, mode=mode)
    results_ed, failures_ed=get_results(df_ed.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'termination', client=client, mode=mode)

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
    for t in eds:
        res=results_ed_s[results_ed_s.target_phoneme.str.contains(t)]
        print(count_values(res[res.gibberish_detected!=res.gibberish_truth].gibberish_detected))
        success_rate = lambda df : len(df[df.gibberish_truth==df.gibberish_detected])/len(df) if len(df)>0 else float('nan')
        print('success rate for '+t, success_rate(res))
    
    return results_v, results_ed, results_ed_s

def test_stress_detection(base_url = 'http://localhost:8000', route='/v2/w2v/stress/', client=app.test_client(), n_ex_by_module=10, mode='v2'):
    df=actor_recordings()
    len_t_seg=df.apply(lambda r: len(r.syllable_parts.split(' ')), axis=1)
    len_p=df.apply(lambda r: len(r.cmu_phonetics.split(' ')), axis=1)
    df[len_t_seg!=len_p]

    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    results_ss, failures_ss=get_results(df_sentence_stress.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'sentence', client=client, mode=mode)
    results_ws, failures_ws=get_results(df_word_stress.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'word', client=client, mode=mode)

    df_test=df_word_stress[df_word_stress.text.str.contains("grandma")].iloc[-1:,:]
    results_ws, failures_ws=get_results(df_test, base_url = base_url, endpoint=route+'word', client=client, mode=mode)

    request_for_audio_file(df_test.iloc[0], base_url = base_url, endpoint=route+'word', client=client, mode=mode)

    return results_ss, failures_ss, results_ws, failures_ws

def test_audio64(base_url = 'http://localhost:8000', client=app.test_client()):
    path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf'
    from utils.text_processing import prefill_for_sentence

    text="I would love to go to ireland"
    r={}
    r['cmu_phonetics']=prefill_for_sentence(text)['cmu_phonetics']
    r['text']=text
    r['audio_file_url']=path
    r['target_phoneme']=float('nan')
    res=request_for_audio_file(r, endpoint='/v2/w2v/stress/sentence', client=client, mode='v2')
    print(res.data)
    assert res.status_code==200
    # res=request_for_audio_file(r, endpoint='/w2v/stress/sentence', client=client, mode='v1')
    # print(res.data)
    # assert res.status_code==200

def pContrast_for_user_data( target_phones='AO1', n_user=10, n_ex_by_ex_type=10):
    user_data=build_user_data_df()
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
    results_df, failures=get_results(df, endpoint='/v2/w2v/contrast/vowel', mode='v2')
    

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

if __name__ == '__main__':
    from test_DL_api import *
    test_actor_recordings(base_url = 'http://localhost:8000', client=requests, n_ex_by_module=10)
    test_stress_detection(base_url = 'http://localhost:8000', client=requests, n_ex_by_module=10)

    test_actor_recordings(base_url = 'http://146.59.241.79:8000', client=requests, n_ex_by_module=10)
    test_stress_detection(base_url = 'http://146.59.241.79:8000', client=requests, n_ex_by_module=10)
    r=test_actor_recordings()
    r=test_stress_detection()

    
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

    from utils.audio_processing import audio64_from_file
    df_sentence_stress[l>50].iloc[0].cmu_phonetics
    audio64=audio64_from_file(df_sentence_stress[l>50].iloc[0].audio_file_url)

    

    # from utils.text_processing import chunk_text
    r=df_sentence_stress[l>50].iloc[0]

    route='/v2/w2v/stress/'
    mode='v2'
    get_results(df_sentence_stress.iloc[3:4,:], endpoint=route+'sentence', client=app.test_client(), mode=mode)
    request_for_audio_file(df_sentence_stress.iloc[3,:], endpoint=route+'sentence', client=app.test_client(), mode=mode)

    from test_DL_api import *;print(pContrast_for_user_data())

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
    