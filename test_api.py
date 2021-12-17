from flask_server import app
from dummy_client import *
import os
os.environ['FLOWSPEECH_KEY']="ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"
import ast
from label_data_processing import get_data_new_content, get_data

def test_api():
    # c=app.test_client()
    with app.test_client() as c:
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
    df_pContrast.index=range(len(df_pContrast))

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

        results['target_phoneme']=df_pContrast['target_phoneme']
        results['alternative_phonemes']=df_pContrast['alternative_phonemes']
        results['audio_file_url']=df_pContrast['audio_file_url']
        
        results[results.status!='success']
        results[results.status!='success'].index.tolist()

        ratio=len(results[results.gibberish_truth==results.gibberish_detected])/len(results)
        results[results.gibberish_truth!=results.gibberish_detected]
        results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('F1')]
        results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('F2')]
        results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('M1')]
        results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('M_')]
        results[results.gibberish_truth!=results.gibberish_detected][results.audio_file_url.str.contains('F_')]

        row=df_pContrast.iloc[results[results.status!='success'].index.tolist()].iloc[2]
        pContrast_for_row(row, url = url, client=client)

        # select phrases for which the target is not the first alternative (which indicates it is a wrong set of alternatives)
        incorrect_alternatives_phrase_ids=df_pContrast[(df_pContrast.target_phoneme!=df_pContrast['alternative_phonemes'].apply(lambda r:ast.literal_eval(r)[0]))
                        &(~df_pContrast.target_phoneme.isin(['D','T','IH0_D']))].fk_phrase_id.unique()
        incorrect_alternatives_phrases=df_pContrast[df_pContrast.fk_phrase_id.isin(incorrect_alternatives_phrase_ids)].drop_duplicates(subset='fk_phrase_id', keep="first")
        
        for p in incorrect_alternatives_phrases.target_phoneme.unique():
            print(p, incorrect_alternatives_phrases[incorrect_alternatives_phrases.target_phoneme==p].fk_phrase_id.tolist())
        
        # select phrases for which there are more than stress variation in the alternatives, which should not happen
        incorrect_alternatives_phrase_ids2=df_pContrast[(df_pContrast.apply(lambda r:len(set([el[-1] for el in ast.literal_eval(r['alternative_phonemes'])])), axis=1)>1)&(~df_pContrast.target_phoneme.isin(['D','T','IH0_D']))].fk_phrase_id.unique()
        incorrect_alternatives_phrases2=df_pContrast[df_pContrast.fk_phrase_id.isin(incorrect_alternatives_phrase_ids2)].drop_duplicates(subset='fk_phrase_id', keep="first")
        
        for p in incorrect_alternatives_phrases2.target_phoneme.unique():
            print(p, incorrect_alternatives_phrases2[incorrect_alternatives_phrases2.target_phoneme==p].fk_phrase_id.tolist())

        return results

def test_particular_cases(url = 'http://localhost:8000/phonemeContrast'):
    
    df=pd.read_csv('data/exercise_data_export.csv')

    # df.target_phoneme.dropna().unique()
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    
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
    rpC=pd.read_csv('results/test_api_phonemeContrast.csv')
    r_ter=pd.read_csv('results/test_api_termination_contrast.csv')

    rpC[~(r_ter==rpC).gibberish_detected][rpC.target_phoneme=='D']
    r_ter[~(r_ter==rpC).gibberish_detected][r_ter.target_phoneme=='D']



if __name__ == '__main__':
    # test_api()
    # test_GE_linguistic_data_content()
    test_GE_linguistic_data_content('http://localhost:8000/termination_contrast')
    test_particular_cases()