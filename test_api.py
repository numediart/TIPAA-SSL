from flask_server import app
from dummy_client import *
import os
os.environ['FLOWSPEECH_KEY']="ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"
import ast

def test_api(c=app.test_client()):
    # c=app.test_client()
    # with app.test_client() as c:
        # res=send_audio_base64(client=c)
        res=send_audio_base64(path='audio_recordings/ended.mp3', client=c)
        assert res.status_code == 200

        res=ast.literal_eval(res.data.decode('utf-8'))
        assert res['status']=='success'
        rID=res['rID']
        print(res)
        # res=call_phoneme_contrast(rID, client=c)
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


def pContrast_for_row(r, url = 'http://localhost:8000/phonemeContrast', client=requests):
    res=send_audio_base64(path='data/scaleway-audio-files/'+r.audio_file_url, client=client)
    assert res.status_code == 200
    res=ast.literal_eval(res.data.decode('utf-8'))
    assert res['status']=='success'
    rID=res['rID']
    print(res)

    res = client.post(url, data={"phonetics":r.cmu_phonetics, 'rID':rID, 
                                'word_idx':str(ast.literal_eval(r.target_word_indexes)[0]), 
                                'syl_idx':str(ast.literal_eval(r.target_syllable_indexes)[0]), 
                                'alternatives':' '.join(ast.literal_eval(r.alternative_phonemes)), 
                                'target':r.target_phoneme})

    return res


def test_GE_linguistic_data_content(url = 'http://localhost:8000/phonemeContrast'):
    df=pd.read_csv('data/exercise_data_export.csv')

    # df.target_phoneme.dropna().unique()
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]

    # client=app.test_client()
    with app.test_client() as client:
        results=[]
        for i,r in df_pContrast.iterrows():
            res=pContrast_for_row(r, url = url, client=client)

    df_pContrast.index=range(len(df_pContrast))

    # df_ed.index=range(len(df_ed))
    # df_v.index=range(len(df_v))

    def get_results(df_pContrast, url = 'http://localhost:8000/phonemeContrast'):
        # client=app.test_client()
        with app.test_client() as client:
            results=[]
            for i,r in df_pContrast.iterrows():
                res=pContrast_for_row(r, url = url, client=client)
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
            # row=df_pContrast.iloc[results[results.status!='success'].index.tolist()].iloc[2]
            # pContrast_for_row(row, url = url, client=client)
        results['target_phoneme']=df_pContrast['target_phoneme']
        results['alternative_phonemes']=df_pContrast['alternative_phonemes']
        results['audio_file_url']=df_pContrast['audio_file_url']
        return results
    
    # row=df_ed[df_ed.text.str.contains('showed')].iloc[0]
    # res=pContrast_for_row(row, url = 'http://ec2-15-188-10-194.eu-west-3.compute.amazonaws.com/phonemeContrast')


    # results=get_results(df_ed[df_ed.text.str.contains('showed')], url=base_url+"/phonemeContrast")

    # df_pContrast=df_pContrast.iloc[424:425]
    # df_pContrast=df_pContrast.iloc[424:425]
    # df_pContrast=df_pContrast.iloc[102:103]
    # df_ed=df_ed[-1:]

    df_pContrast.words.apply(lambda r:len(r))
    df_pContrast[-1:].cmu_phonetics
    
    # results=get_results(df_pContrast, url=base_url+"/phonemeContrast")
    results=get_results(df_pContrast, url=url)
    # df_ed=df_ed[-2:]
    # results_ed=get_results(df_ed, url=base_url+"/final_phoneme")
    # results_v=get_results(df_v, url=base_url+"/phonemeContrast")

    
    # results_v[results_v.status!='success']
    # results_ed[results_ed.status!='success']

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

def test_particular_cases(url = 'http://localhost:8000/phonemeContrast'):
    
    df=pd.read_csv('data/exercise_data_export.csv')

    # df.target_phoneme.dropna().unique()
    # those who don't have NaN in target
    # df_pContrast=df.loc[df.target_phoneme.dropna().index]
    
    # client=app.test_client()

    # indices selected of historical errors from above function that are therefore interesting test cases 
    # selected_idxs=[1524] 
    selected_idxs=[968, 1123, 1524] 
    with app.test_client() as client:
        results=[]
        for idx in selected_idxs:
            row=df.iloc[idx]
            res=pContrast_for_row(row, url = url, client=client)
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
        return results


def result_analysis():
    
    rpC=test_GE_linguistic_data_content('http://localhost:8000/phonemeContrast')
    r_ter=test_GE_linguistic_data_content('http://localhost:8000/termination_contrast')

    
    rpC.to_csv('results/test_api_phonemeContrast.csv')
    r_ter.to_csv('results/test_api_termination_contrast.csv')

    # rpC=pd.read_csv('results/test_api_phonemeContrast.csv')
    # r_ter=pd.read_csv('results/test_api_termination_contrast.csv')

    len(rpC[~(r_ter==rpC).gibberish_detected])

    rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme=='D']
    r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme=='D']
    rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme=='T']
    r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme=='T']

    
    rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme.str.contains('AO')]
    r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme.str.contains('AO')]
    rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme.str.contains('OW')]
    r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme.str.contains('OW')]
    rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme.str.contains('IH')]
    r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme.str.contains('IH')]
    rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme.str.contains('IY')]
    r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme.str.contains('IY')]



if __name__ == '__main__':
    # test_api()
    # test_GE_linguistic_data_content()
    rpC=test_GE_linguistic_data_content('http://localhost:8000/phonemeContrast')
    r_ter=test_GE_linguistic_data_content('http://localhost:8000/termination_contrast')
    test_particular_cases()
