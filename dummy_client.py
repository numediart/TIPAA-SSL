import requests
import json
from text_processing import phonetics_from_sentence, prefill_for_sentence
import pandas as pd
import os
syllables_data=pd.read_csv('data/syllables.csv')

from label_data_processing import get_data
import time
import base64
import ast
from tqdm import tqdm



def send_audio(path='audio_recordings/WS_111_toothpaste.wav', base_url = 'http://localhost:8000', client=requests):
    url=base_url+"/upload"
    # print(url)
    with open(path, 'rb') as file:
        files = {'file': file}
        req = client.post(url, files=files)
    # This is for compatibility between requests module and flask's test_client
    if '_content' in res.__dict__.keys(): res.data=res._content
    return req

def call_vowel_stresses(rID, text='I would love to go to ireland !', base_url = 'http://localhost:8000', client=requests):
    url=base_url+"/vowel_stresses"
    phonetics=phonetics_from_sentence(text)
    res = client.post(url, data={"phonetics":json.dumps(phonetics), 'rID':rID})
    
    # This is for compatibility between requests module and flask's test_client
    if '_content' in res.__dict__.keys(): res.data=res._content
    
    return res

def call_module(rID, text='I would love to go to ireland!', module='sentenceStress', base_url = 'http://localhost:8000', client=requests):
    url=base_url+"/flowspeech/"
    d=prefill_for_sentence(text, syllables_data)
    phonetics=d['cmu_phonetics']
    # res = client.post(url+module, data={"phonetics":json.dumps(phonetics), 'rID':rID})
    res = client.post(url+module, data={"text":text, "phonetics":phonetics, 'rID':rID})

    # This is for compatibility between requests module and flask's test_client
    if '_content' in res.__dict__.keys(): res.data=res._content
    
    return res

def call_phoneme_contrast(rID, text='turned around', word_idx=0, syl_idx=0, target='D', alternatives="T D IH0_D", base_url = 'http://localhost:8000', client=requests):
    url=base_url+"/phonemeContrast"
    d=prefill_for_sentence(text, syllables_data)
    phonetics=d['cmu_phonetics']
    # phonetics=phonetics_from_sentence(text)
    # res = client.post(url, data={"phonetics":json.dumps(phonetics), 'rID':rID, 'word_idx':word_idx, 'alternatives':alternatives, 'target':target})
    res = client.post(url, data={"phonetics":phonetics, 'rID':rID, 'word_idx':word_idx, 'syl_idx':syl_idx, 'alternatives':alternatives, 'target':target})
    
    # This is for compatibility between requests module and flask's test_client
    if '_content' in res.__dict__.keys(): res.data=res._content
    
    return res

def call_prefill_for_sentence(sentence, base_url = 'http://localhost:8000', client=requests):
    url=base_url+"/prefill_from_phrase"
    res = client.post(url, data={"phrase":sentence})
    
    # This is for compatibility between requests module and flask's test_client
    if '_content' in res.__dict__.keys(): res.data=res._content
    
    return res

def send_audio_base64(path='audio_recordings/WS_111_toothpaste.wav', base_url = 'http://localhost:8000', client=requests):
    # based on :
    # https://stackoverflow.com/questions/50279380/how-to-decode-base64-string-directly-to-binary-audio-format
    encode_string = base64.b64encode(open(path, "rb").read())
    res = client.post(base_url+"/send_base64_audio", data={"audio":encode_string, "API_KEY":"ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"})

    # This is for compatibility between requests module and flask's test_client
    if '_content' in res.__dict__.keys(): res.data=res._content
    
    return res

def crash_test(base_url="http://ec2-13-37-107-52.eu-west-3.compute.amazonaws.com/", client=requests):
    rIDs=[]
    print("uploads starting")
    for i in tqdm(range(100)):
        res=send_audio_base64(path='audio_recordings/turned_around.mp3', base_url=base_url, client=client)
        res=ast.literal_eval(res.data.decode('utf-8'))
        rID=res['rID']
        rIDs.append(rID)
    
    print("phoneme contrast calls starting")
    for rID in rIDs:
        call_phoneme_contrast(rID, base_url=base_url, client=client)
    
    audio_path="../audio-with-analysis-ids/audio/"
    d=get_data()
    focusType='sentencestress'
    d=d[d.focusType==focusType]

    # send all the audios (n times) and get rIDs 
    rIDs=[]
    n=2
    start=time.time()
    for _ in range(n):
        for i,row in d.iterrows():
            path=os.path.join(audio_path, row.primaryKey+'.wav')
            res=send_audio_base64(path=path, base_url=base_url)
            rID=res.data
            rIDs.append(rID)
    avg_duration=(time.time()-start)/(len(d)*n)
    print('avg duration upload:', avg_duration)

    start=time.time()
    tot_d=pd.concat([d]*n)
    tot_d['rID']=rIDs
    for i,row  in tot_d.iterrows():
        call_module(row['rID'].decode('utf-8'), text=row.text, module='sentenceStress', base_url = base_url, client=client)
    avg_duration=(time.time()-start)/(len(d)*n)
    print('avg duration processing sentenceStress:', avg_duration)

# deprecated functions
if False:
    def call_module(module='sentenceStress', filename='SS_1_i_would_love_to_go_to_ireland.wav', sentenceID=1, client=requests):
        res = client.post('http://localhost:8000/flowspeech/'+module, data={"sentenceID":str(sentenceID), 'filename':filename})
        print(res.data)
        return res.data

if __name__ == "__main__":
    res=send_audio_base64(path='audio_recordings/turned_around.mp3').data
    rID=ast.literal_eval(res.decode('utf-8'))['rID']
    res=call_phoneme_contrast(rID)
    res.data

    res=send_audio(path='audio_recordings/turned_around.mp3')
    rID1=res.data.decode('utf-8')
    res=call_phoneme_contrast(rID)
    res.data

    res=send_audio(path='audio_recordings/turned_around.mp3', base_url="http://ec2-13-37-107-52.eu-west-3.compute.amazonaws.com")
    rID=res.data.decode('utf-8')

    res=send_audio_base64(path='audio_recordings/turned_around.mp3', base_url="http://ec2-13-37-107-52.eu-west-3.compute.amazonaws.com").data
    rID=ast.literal_eval(res.decode('utf-8'))['rID']

    call_phoneme_contrast( rID, base_url="http://ec2-13-37-107-52.eu-west-3.compute.amazonaws.com")
    
    call_prefill_for_sentence( "Kayla isn't angry at Tyler", base_url="http://ec2-13-37-107-52.eu-west-3.compute.amazonaws.com")
    call_prefill_for_sentence("Kayla isn't angry at Tyler")
    
    res=send_audio(path='audio_recordings/I_visited_italy.mp3')
    rID2=res.data
    call_phoneme_contrast(rID2.decode('utf-8'), text='I visited italy', word_idx=1, syl_idx=1, target='IH0', alternatives="IH0 IY0")

    
    res=send_audio(path='audio_recordings/I_visited_italy.mp3', base_url="http://ec2-13-37-107-52.eu-west-3.compute.amazonaws.com")
    rID2=res.data
    call_phoneme_contrast(rID2.decode('utf-8'), text='I visited italy', word_idx=1, syl_idx=1, target='IH0', alternatives="IH0 IY0", base_url="http://ec2-13-37-107-52.eu-west-3.compute.amazonaws.com")
    
    res=send_audio(path='audio_recordings/I_visited_italy.mp3')
    rID=res.data
    call_phoneme_contrast(rID.decode('utf-8'), text='I visited italy', word_idx=1, syl_idx=2, target='IH0_D', alternatives="T D IH0_D", base_url="http://ec2-13-37-107-52.eu-west-3.compute.amazonaws.com")

    res=send_audio(path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    rID=res.data
    call_vowel_stresses(rID.decode('utf-8'))
    
    res=send_audio_base64(path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav').data
    rID=ast.literal_eval(res.decode('utf-8'))['rID']
    call_module(rID)

    
    res=send_audio(path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    rID=res.data
    call_module(rID.decode('utf-8'), module="wordStress")

    res=send_audio(path='audio_recordings/WS_111_toothpaste.wav')
    rID=res.data
    call_module(rID.decode('utf-8'), text="toothpaste", module="wordStress")

    # res=send_audio(path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    # rID=res.data
    # call_sentenceStress(rID.decode('utf-8'))

    res=send_audio(path='audio_recordings/iC_111_slip.wav')
    rID=res.data
    # call_module(rID, filename='iC_111_slip.wav', module="iContrast")
    # call_vowel_stresses()
    call_vowel_stresses(rID, text='sleep')

    send_audio('audio_recordings/turned_around.mp3')
    call_phoneme_contrast()

    send_audio('audio_recordings/Laaw_MP3.mp3')
    alternatives="['AO0', 'OW0','AO1', 'OW1','AO2', 'OW2']"
    call_phoneme_contrast(filename='Laaw_MP3.mp3', text='law',  word_idx=0, target='AO1', alternatives=alternatives)

    send_audio('audio_recordings/iC_112_leave.wav')
    alternatives="['IH0', 'IY0','IH1', 'IY1','IH2', 'IY2']"
    call_phoneme_contrast(filename='iC_112_leave.wav', text='leave',  word_idx=0, target='IY1', alternatives=alternatives)
    
    send_audio('audio_recordings/ed_acceptEED.wav')
    # alternatives="['IH0', 'IY0','IH1', 'IY1','IH2', 'IY2']"
    alternatives="['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']"
    # alternatives="['T', 'D', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']"
    # alternatives="['T', 'D']"
    call_phoneme_contrast(filename='ed_acceptEED.wav', text='accepted',  word_idx=0, target="IH0 D", alternatives=alternatives)
