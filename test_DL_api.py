from flask_server import app
from dummy_client import *
import os
os.environ['FLOWSPEECH_KEY']="ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"
import ast
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from test_api import get_results, request_for_audio_file
from utils.label_data_processing import actor_recordings


def get_results(df_pContrast, base_url = 'http://localhost:8000', endpoint='/w2v/contrast/vowel', client=app.test_client()):
    df_pContrast.index=range(len(df_pContrast))
    results_records=[]
    failures=[]
    # r=df_pContrast.iloc[4]
    print(len(df_pContrast))
    for i,r in tqdm(df_pContrast.iterrows()):
        res=request_for_audio_file(r, base_url = base_url, endpoint=endpoint, client=client)
        if 'success' in res.data.decode('utf-8'):
            d=ast.literal_eval(res.data.decode('utf-8'))
            d['cmu_phonetics']=r.cmu_phonetics
            d['audio_file_url']=r.audio_file_url
            d['text']=r.text
            d['target_phoneme']=r.target_phoneme
            results_records.append(d)
        else:
            failures.append(res)
    results_df=pd.DataFrame.from_records(results_records)
    return results_df, failures


def test_GE_linguistic_data_content(base_url = 'http://localhost:8000', client=app.test_client(), n_ex_by_module=10):
    df=actor_recordings()

    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    vowels=['IH','IY','OW','AO','AA']
    eds=['T','D','IH0_D']

    df_v=df_pContrast[~df_pContrast.target_phoneme.isin(eds)]
    df_ed=df_pContrast[df_pContrast.target_phoneme.isin(eds)]

    route='/w2v/contrast/'
    results_v, failures_v=get_results(df_v.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'vowel', client=client)
    results_ed, failures_ed=get_results(df_ed.iloc[:n_ex_by_module,:], base_url = base_url, endpoint=route+'syllable', client=client)

    results_v[results_v.gibberish_truth!=results_v.gibberish_detected]
    results_ed[results_ed.gibberish_truth!=results_ed.gibberish_detected]
    
    for v in vowels:
        res=results_v[results_v.target_phoneme.str.contains(v)]
        errors=results_v[results_v.gibberish_truth!=results_v.gibberish_detected][results_v.target_phoneme.str.contains(v)&~results_v.target_phoneme.str.contains('_')]
        if len(res)>0:
            print('Target is', v, 'error rate:', len(errors)/len(res), 'n:', len(res))
            confusion_ps=errors.phonetic_detection.unique()
            for p in confusion_ps:
                error_part=res[res.phonetic_detection==p]
                print(p, len(error_part)/len(res))


def test_stress_detection(base_url = 'http://localhost:8000', client=app.test_client(), n_ex_by_module=10):
    df=actor_recordings()

    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    
    results_ss, failures_ss=get_results(df_sentence_stress.iloc[:n_ex_by_module,:], base_url = base_url, endpoint='/w2v/stress/sentence', client=client)
    results_ws, failures_ws=get_results(df_word_stress.iloc[:n_ex_by_module,:], base_url = base_url, endpoint='/w2v/stress/word', client=client)



# from utils.label_data_processing import exercise_data, build_user_data_df
from utils.label_data_processing import build_user_data_df
exercise_data=pd.read_csv('data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv') 

def pContrast_for_user_data( target_phones='AO1', n_user=50):
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
            selections.append(selection[selection.user_id==u][selection.exercise_id==ex])
    df=pd.concat(selections)

    results=get_results(df)

    df['phonetic_detection']=results['phonetic_detection']
    df['status']=results['status']
    df.to_csv('performance_results/vowel_contrast_user_data_'+target_phones+'.csv') 

    d= results.phonetic_detection.value_counts()
    d=d / d.sum()*100

    return df, d



if __name__ == '__main__':
    # test_api()
    r=test_GE_linguistic_data_content()
    
    # idx=r[r.status!='success'].index[0]
    # row=df_pContrast.loc[idx]
    # request_for_audio_file(row, endpoint='/w2v/contrast/phoneme', client=app.test_client())

    import requests
    # try with local docker container
    test_stress_detection(client=requests)
    test_GE_linguistic_data_content(client=requests)

    df, d=pContrast_for_user_data()
    print(d)
    