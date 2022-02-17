from flask_server import app
from dummy_client import *
import os
os.environ['FLOWSPEECH_KEY']="ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"
import ast
from tqdm import tqdm
import shutil
import numpy as np
import soundfile as sf

def make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def test_api(c=app.test_client()):
    res=send_audio_base64(path='audio_recordings/ended.mp3', client=c)
    assert res.status_code == 200

    res=ast.literal_eval(res.data.decode('utf-8'))
    assert res['status']=='success'
    rID=res['rID']
    print(res)
    res=call_phoneme_contrast(rID, text="ended", word_idx=0, syl_idx=1, target="IH0_D", alternatives="T D IH0_D", client=c)

    assert res.status_code == 200

    res=send_audio_base64(path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', client=c)

    assert res.status_code == 200
    res=ast.literal_eval(res.data.decode('utf-8'))
    rID=res['rID']
    res=call_module(rID, fake_mistake=True, client=c)

    base_url = 'http://localhost:8000'
    crash_test(base_url=base_url, client=c)

    # assert c.get('http://localhost:8000/').status_code == 200
    # assert send_audio(client=c).status_code==200
    # assert eval(call_module(client=c).data)['status']=='success'
    # assert eval(call_phoneme_contrast(client=c).data)['status']=='success'
    # assert eval(call_module(module='wordStress', filename='WS_111_toothpaste.wav', sentenceID=111, client=c).data)['status']=='success'


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

    res=send_audio_base64(path=r['audio_file_url'], base_url = base_url, client=client)
    assert res.status_code == 200
    # This is for compatibility between requests module and flask's test_client
    if client==requests: res.data=res._content
    res=ast.literal_eval(res.data.decode('utf-8'))
    assert res['status']=='success'
    rID=res['rID']
    print(res)

    if 'Stress' in endpoint:
        res = client.post(url, data={"phonetics":r['cmu_phonetics'],"text":r['text'], 'rID':rID, 
                                    'target':r['target_phoneme']})
    else:
        res = client.post(url, data={"phonetics":r['cmu_phonetics'], 'rID':rID, 
                                    'word_idx':str(ast.literal_eval(r['target_word_indexes'])[0]), 
                                    'syl_idx':str(ast.literal_eval(r['target_syllable_indexes'])[0]), 
                                    'target':r['target_phoneme']})
    if client==requests: res.data=res._content
    return res

def get_results(df_pContrast, base_url = 'http://localhost:8000', endpoint='/phonemeContrast', client=app.test_client()):
    # client=app.test_client()
    # with app.test_client() as client:
    results=[]
    for i,r in tqdm(df_pContrast.iterrows()):
        res=request_for_audio_file(r, base_url = base_url, endpoint=endpoint, client=client)
        if '''"phonetic_detection":''' in res.data.decode('utf-8'):
            result=ast.literal_eval(res.data.decode('utf-8'))
        else:
            result={}
            result['status']=res.data.decode('utf-8')
            result['phonetic_detection']=''
            result['gibberish_truth']=''
            result['gibberish_detected']=''
        results.append(result)
    
    results=pd.DataFrame.from_records(results)
    results.index=df_pContrast.index
    # row=df_pContrast.iloc[results[results.status!='success'].index.tolist()].iloc[2]
    # request_for_audio_file(row, url = url, client=client)
    results['target_phoneme']=df_pContrast['target_phoneme']
    # results['alternative_phonemes']=df_pContrast['alternative_phonemes']
    results['audio_file_url']=df_pContrast['audio_file_url']
    return results

def test_GE_linguistic_data_content(base_url = 'http://localhost:8000', endpoint='/phonemeContrast', client=app.test_client()):
    df=pd.read_csv('data/exercise_data_export.csv')
    df['audio_file_url']='data/scaleway-audio-files/'+df['audio_file_url']

    # df.target_phoneme.dropna().unique()
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]

    results=[]
    # for i,r in df_pContrast.iterrows():
    #     res=request_for_audio_file(r, base_url = base_url, endpoint=endpoint, client=client)

    df_pContrast.index=range(len(df_pContrast))

    # df_ed.index=range(len(df_ed))
    # df_v.index=range(len(df_v))

    df_pContrast.words.apply(lambda r:len(r))
    df_pContrast[-1:].cmu_phonetics
    
    # results=get_results(df_pContrast, url=base_url+"/phonemeContrast", client=client)
    results=get_results(df_pContrast, base_url = base_url, endpoint=endpoint, client=client)
    # df_ed=df_ed[-2:]
    # results_ed=get_results(df_ed, url=base_url+"/final_phoneme", client=client)
    # results_v=get_results(df_v, url=base_url+"/phonemeContrast", client=client)

    results[results.status!='success']
    results[results.status!='success'].index.tolist()

    results[~results.status.str.contains('success')]
    # results[~results.status.str.contains('success')].iloc[0].status
    
    vowels=['IH','IY','OW','AO','AA']
    eds=['T','D','IH0_D']

    mask=results.apply(lambda r:sum([str(i)== r.target_phoneme[-1] for i in range(3)]), axis=1)
    results_ed=results[~mask.astype(bool)]
    results_vowels=results[mask.astype(bool)]

    # ratio_vowels=len(results_vowels[results_vowels.gibberish_truth==results_vowels.gibberish_detected])/len(results_vowels)
    # ratio_ed=len(results_ed[results_ed.gibberish_truth==results_ed.gibberish_detected])/len(results_ed)

    # results[results.gibberish_truth!=results.gibberish_detected]
    # results[results.target_phoneme.str.contains('IY')]
    # results[results.target_phoneme.str.contains('OW')]

    for v in vowels:
        res=results_vowels[results_vowels.target_phoneme.str.contains(v)]
        errors=results_vowels[results_vowels.gibberish_truth!=results_vowels.gibberish_detected][results_vowels.target_phoneme.str.contains(v)&~results_vowels.target_phoneme.str.contains('_')]
        print('Target is', v, len(errors)/len(res), 'n:', len(res))
        confusion_ps=errors.phonetic_detection.unique()
        for p in confusion_ps:
            error_part=res[res.phonetic_detection==p]
            print(p, len(error_part)/len(res))
    
    for term in eds:
        res=results_ed[results_ed.target_phoneme==term]
        errors=results_ed[results_ed.gibberish_truth!=results_ed.gibberish_detected][results_ed.target_phoneme==term]
        print('Target is', term, len(errors)/len(res))
        confusion_ps=errors.phonetic_detection.unique()
        for p in confusion_ps:
            error_part=res[res.phonetic_detection==p]
            print(p, len(error_part)/len(res))
    
    results[results.gibberish_truth!=results.gibberish_detected][results.target_phoneme.str.contains('IH')&~results.target_phoneme.str.contains('_')]
    results[results.gibberish_truth!=results.gibberish_detected][results.target_phoneme.str.contains('IY')]
    results[results.gibberish_truth!=results.gibberish_detected][results.target_phoneme.str.contains('OW')]
    results[results.gibberish_truth!=results.gibberish_detected][results.target_phoneme.str.contains('AO')]
    results[results.gibberish_truth!=results.gibberish_detected][results.target_phoneme.str.contains('AA')]

    results[results.gibberish_truth!=results.gibberish_detected][results.target_phoneme=='D']
    results[results.gibberish_truth!=results.gibberish_detected][results.target_phoneme=='T']
    results[results.gibberish_truth!=results.gibberish_detected][results.target_phoneme=='IH0_D']
    
    results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('F1')]
    results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('F2')]
    results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('M1')]
    results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('M_')]
    results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('F_')]

    return results

def test_particular_cases(base_url = 'http://localhost:8000', endpoint='/phonemeContrast', client=app.test_client()):
    
    df=pd.read_csv('data/exercise_data_export.csv')
    df['audio_file_url']='data/scaleway-audio-files/'+df['audio_file_url']

    # df.target_phoneme.dropna().unique()
    # those who don't have NaN in target
    # df_pContrast=df.loc[df.target_phoneme.dropna().index]
    
    # client=app.test_client()

    # indices selected of historical errors from above function that are therefore interesting test cases 
    # selected_idxs=[1524] 
    selected_idxs=[968, 1123, 1524] 
    # with app.test_client() as client:
    df_selected_idx=df.iloc[selected_idxs, :]

    results=get_results(df_selected_idx, base_url = base_url, endpoint=endpoint, client=client)

    return results

def test_termination_contrast():
    test_GE_linguistic_data_content(endpoint='/terminationContrast')

def test_user_recordings(base_url = 'http://localhost:8000', client=app.test_client(), n_audios=1, n_ex_by_ex_type=1, all_results_path='performance_results/user_recording_all_results.csv'):
    """[summary]

    Args:
        base_url (str, optional): [description]. Defaults to 'http://localhost:8000'.
        client ([type], optional): [description]. Defaults to app.test_client().
        n_audios (int, optional): number of audios to randomly pick and save in a folder. Defaults to 20.
        n_ex_by_ex_type (int, optional): number of exericises selected for each exercise type. Defaults to 5.

    Returns:
        [type]: [description]
    """
    from user_audio_data import df_errors, exs_sort_by_n_errors, n_errors, exercise_data
    # "cmu_phonetics", "target_word_indexes", 
    # "target_syllable_indexes", "alternative_phonemes", "target_phoneme"

    endpoint_dict={
        'SENTENCE_STRESS':'/flowspeech/sentenceStress',
        'WORD_STRESS':'/flowspeech/wordStress',
        'FINAL_ED':'/terminationContrast',
        'VOWEL_CONTRAST':'/phonemeContrast'
    }

    all_results=[]
    all_dfs=[]
    # for module_type in df_errors.module_type.unique()[1:2]:
    for module_type in endpoint_dict.keys():
        print('module '+module_type, len(df_errors[df_errors.module_type==module_type]))
        # module_type='VOWEL_CONTRAST'
        df_vc=df_errors[df_errors.module_type==module_type]
        records=[]
        # for ex in exs_sort_by_n_errors[:n_worst]:
        for ex in exs_sort_by_n_errors:
            if ex in df_vc.exercise_id.unique():
                for i,r in df_vc[df_vc.exercise_id==ex][:n_ex_by_ex_type].iterrows():
                    # even though the first line has the same result, it is important do do it here to have
                    # different copies modified afterwards. 
                    req_data=exercise_data[exercise_data.exercise_id==ex].iloc[0].to_dict()
                    req_data["audio_file_url"]=r.fpath
                    # print(req_data["audio_file_url"])
                    records.append(req_data)
        df_records=pd.DataFrame.from_records(records)
        print(df_records)
        # those who don't have NaN in target
        if module_type in ['FINAL_ED', 'VOWEL_CONTRAST']:
            df_records=df_records.loc[df_records.target_phoneme.dropna().index]

        all_dfs.append(df_records)
        df_records['size']=df_records.apply(lambda r:os.path.getsize(r.audio_file_url), axis=1)

        # Here I filter examples to be smaller than 400KB, because above, I can have an error 413 "Request Entity Too Large". Not sure exactly where it
        # comes from
        df_records=df_records[df_records['size']<400*1024]

        results=get_results(df_records, base_url = base_url, endpoint=endpoint_dict[module_type], client=client)
        results=pd.concat([results,df_records], axis=1)
        all_results.append(results)
    
    # all_dfs=pd.concat(all_df)

    # for results, module_type in zip(all_results, df_errors.module_type.unique()[1:2]):
    for results, module_type in zip(all_results, df_errors.module_type.unique()):
        # for each type of message, take n_audios randomly picked
        dfs=[]
        for status in results.status.unique():
            df=results[results.status==status].sample(frac = 1)[:n_audios]
            if "DOCTYPE HTML PUBLIC" in status: status="Internal_Server_Error"
            path='performance_results/user_data_analysis_terminationContrast/'+module_type+'/'+status.replace(' ','_').replace(':','').replace('/','')
            make_dir(path)
            
            df.to_csv(path+'/data.csv')
            for i,r in df.iterrows():
                shutil.copy(r.audio_file_url.iloc[0], path)
            dfs.append(df)

    all_results_df=pd.concat(all_results)
    all_results_df.to_csv(all_results_path)

    return all_results_df

def detection_tests(all_results_path='performance_results/user_recording_all_results.csv'):
    all_results=pd.read_csv(all_results_path)
    from DL_speech_tech import run_VAD, load_audio_with_preprocessing, speech_enhancement
    probs=[]
    for i,r in tqdm(all_results.iterrows()):
        s,fs=load_audio_with_preprocessing(r[:-1].audio_file_url)

        # I wanted to try after speech enhancement, but the problem is we don't know the influence it will have on the VAD
        # which is trained on noisy speech. After a quick look, it does not seem better
        # s=speech_enhancement(s)
        prob=run_VAD(s).numpy().flatten()
        probs.append(prob)
    all_results["VAD probs"]=probs
    all_results["VAD max prob"]=all_results["VAD probs"].apply(lambda r: max(r))

    np.histogram(all_results["VAD max prob"])

    selection_low_VAD=all_results[all_results["VAD max prob"]<0.3]
    selection_low_VAD[selection_low_VAD.status.str.contains('pitch')]
    uncatched_low_VAD=selection_low_VAD[~selection_low_VAD.status.str.contains('pitch')&~selection_low_VAD.status.str.contains('audio')]
    path='performance_results/low_VAD/'
    make_dir(path)
    for i,r in uncatched_low_VAD.iterrows():
        shutil.copy(r.audio_file_url, path)


    # all_results.to_csv(all_results_path)

def user_recordings_results_analysis(all_results_path='performance_results/user_recording_all_results.csv'):

    all_results=pd.read_csv(all_results_path)

    all_results[all_results.module_type=='SENTENCE_STRESS']
    all_results[all_results.module_type=='WORD_STRESS']
    all_results[all_results.module_type=='FINAL_ED']
    all_results[all_results.module_type=='VOWEL_CONTRAST']

    all_results[all_results.status.str.contains('success')]

    all_results_vc=all_results[all_results.module_type=='VOWEL_CONTRAST']
    all_results_vc[all_results_vc.status.str.contains('empty')]

    all_results_fed=all_results[all_results.module_type=='FINAL_ED']
    all_results_fed[all_results_fed.status.str.contains('expected ph')]

    select_errors=all_results[~all_results.status.str.contains('success')]
    select_errors[select_errors.module_type=='WORD_STRESS']
    for i,r in select_errors[select_errors.module_type=='WORD_STRESS'].iterrows():
        request_for_audio_file(r[:-1], base_url = 'http://localhost:8000', endpoint='/flowspeech/wordStress', client=app.test_client())
    
    return all_results

def get_server_and_local_results():
    res={}

    res['rpC_dev']=test_GE_linguistic_data_content('https://dev-speech-processing.flowchase.app', endpoint='/phonemeContrast', client=requests)
    res['r_ter_dev']=test_GE_linguistic_data_content('https://dev-speech-processing.flowchase.app', endpoint='/terminationContrast', client=requests)

    res['rpC_prod']=test_GE_linguistic_data_content('https://speech-processing.flowchase.app', endpoint='/phonemeContrast', client=requests)
    res['r_ter_prod']=test_GE_linguistic_data_content('https://speech-processing.flowchase.app', endpoint='/terminationContrast', client=requests)

    res['rpC_local']=test_GE_linguistic_data_content('http://localhost:8000', endpoint='/phonemeContrast')
    res['r_ter_local']=test_GE_linguistic_data_content('http://localhost:8000', endpoint='/terminationContrast')

    for k in res: res[k].to_csv('performance_results/'+k+'.csv')

def server_comparison_analysis():
    
    # rpC=test_GE_linguistic_data_content('http://localhost:8000/phonemeContrast')
    # r_ter=test_GE_linguistic_data_content('http://localhost:8000/terminationContrast')

    # rpC.to_csv('performance_results/test_api_phonemeContrast.csv')
    # r_ter.to_csv('performance_results/test_api_terminationContrast.csv')

    
    ks=['rpC_dev', 'r_ter_dev', 'rpC_prod', 'r_ter_prod', 'rpC_local', 'r_ter_local']
    res={}
    for k in ks: res[k]=pd.read_csv('performance_results/'+k+'.csv')

    # checking nans are either "success: all of the elements were out of vocab" or "success: phonetic detection is empty"
    res['rpC_dev'][res['rpC_dev'].isnull().any(1)]

    for k in ks: res[k]=pd.read_csv('performance_results/'+k+'.csv').dropna()

    (res['rpC_dev']==res['rpC_prod']).sum()
    (res['rpC_dev']==res['rpC_local']).sum()

    (res['rpC_dev']==res['r_ter_dev']).sum()
    (res['r_ter_dev']==res['r_ter_local']).sum()

    (res['rpC_local']==res['r_ter_local']).sum()
    (res['rpC_local']!=res['r_ter_local']).sum()

    res['rpC_local'][~(res['r_ter_local']==res['rpC_local']).gibberish_detected]#[res['rpC_local'].target_phoneme=='D']
    res['r_ter_local'][~(res['r_ter_local']==res['rpC_local']).gibberish_detected]#[res['rpC_local'].target_phoneme=='D']

    
    res['rpC_dev'][~(res['r_ter_dev']==res['rpC_dev']).gibberish_detected]#[res['rpC_local'].target_phoneme=='D']
    res['r_ter_dev'][~(res['r_ter_dev']==res['rpC_dev']).gibberish_detected]#[res['rpC_local'].target_phoneme=='D']

    # rpC=pd.read_csv('performance_results/test_api_phonemeContrast.csv')
    # r_ter=pd.read_csv('performance_results/test_api_termination_contrast.csv')

    # len(rpC[~(r_ter==rpC).gibberish_detected])

    # rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme=='D']
    # r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme=='D']
    # rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme=='T']
    # r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme=='T']

    
    # rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme.str.contains('AO')]
    # r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme.str.contains('AO')]
    # rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme.str.contains('OW')]
    # r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme.str.contains('OW')]
    # rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme.str.contains('IH')]
    # r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme.str.contains('IH')]
    # rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme.str.contains('IY')]
    # r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme.str.contains('IY')]



if __name__ == '__main__':
    # test_api()
    # test_GE_linguistic_data_content()
    # rpC=test_GE_linguistic_data_content('http://localhost:8000/phonemeContrast')

    # rpC=test_GE_linguistic_data_content('https://dev-speech-processing.flowchase.app/phonemeContrast')
    # r_ter=test_GE_linguistic_data_content('http://localhost:8000/terminationContrast')
    all_results=test_user_recordings(base_url = 'http://localhost:8000', client=app.test_client(), n_audios=20, n_ex_by_ex_type=5, all_results_path='performance_results/user_recording_all_results_local.csv')
    # all_results=test_user_recordings(base_url = 'http://localhost:8000', client=app.test_client(), n_audios=20, n_ex_by_ex_type=5, all_results_path='performance_results/user_recording_all_results_SE.csv')
    test_user_recordings()
    all_results=test_user_recordings(base_url = 'https://dev-speech-processing.flowchase.app', client=requests, n_audios=20, n_ex_by_ex_type=5, all_results_path='performance_results/user_recording_all_results_dev.csv')

    rpC_dev=test_GE_linguistic_data_content('https://dev-speech-processing.flowchase.app', endpoint='/phonemeContrast', client=requests)
    r_ter_dev=test_GE_linguistic_data_content('https://dev-speech-processing.flowchase.app', endpoint='/terminationContrast', client=requests)

    rpC_prod=test_GE_linguistic_data_content('https://speech-processing.flowchase.app', endpoint='/phonemeContrast', client=requests)
    r_ter_prod=test_GE_linguistic_data_content('https://speech-processing.flowchase.app', endpoint='/terminationContrast', client=requests)

    rpC_local=test_GE_linguistic_data_content('http://localhost:8000', endpoint='/phonemeContrast')
    r_ter_local=test_GE_linguistic_data_content('http://localhost:8000', endpoint='/terminationContrast')
    
    test_GE_linguistic_data_content()

    test_particular_cases('https://dev-speech-processing.flowchase.app', endpoint='/phonemeContrast', client=requests)
    test_particular_cases()


    base_url='https://dev-speech-processing.flowchase.app'
    endpoint='/phonemeContrast'
    url=base_url+endpoint
