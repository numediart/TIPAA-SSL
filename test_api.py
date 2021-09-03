from flask_server import app
from dummy_client import *
import requests
from text_processing import phonetics_from_sentence
import json
import base64
import os
os.environ['FLOWSPEECH_KEY']="ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"
import ast

# def call_module(module='sentenceStress', filename='SS_1_i_would_love_to_go_to_ireland.wav', sentenceID=1, client=requests):
#     res = client.post('http://localhost:5000/flowspeech/'+module, data={"sentenceID":str(sentenceID), 'filename':filename})
#     print(res)
#     return res

# def send_audio_base64(path='audio_recordings/WS_111_toothpaste.wav', base_url = 'http://localhost:8000', client=requests):
#     # based on :
#     # https://stackoverflow.com/questions/50279380/how-to-decode-base64-string-directly-to-binary-audio-format
#     encode_string = base64.b64encode(open(path, "rb").read())
#     res = client.post(base_url+"/send_base64_audio", data={"audio":encode_string, "API_KEY":"ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"})
#     return res

# def call_phoneme_contrast(rID, text='turned around', word_idx=0, target='D', alternatives="['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']", base_url = 'http://localhost:8000', client=requests):
#     phonetics=phonetics_from_sentence(text)
#     res = client.post(base_url+'/phonemeContrast', data={"phonetics":json.dumps(phonetics), 'rID':rID, 'word_idx':word_idx, 'alternatives':alternatives, 'target':target})
#     return res

def test_api():
    # c=app.test_client()
    with app.test_client() as c:
        res=send_audio_base64(client=c)
        res=ast.literal_eval(res.data.decode('utf-8'))
        assert res['status']=='success'
        rID=res['rID']
        print(res)
        call_phoneme_contrast(rID, client=c).data
        
        base_url = 'http://localhost:8000'
        crash_test(base_url=base_url, client=c)

        # assert c.get('http://localhost:8000/').status_code == 200
        # assert send_audio(client=c).status_code==200
        # assert eval(call_module(client=c).data)['status']=='success'
        # assert eval(call_phoneme_contrast(client=c).data)['status']=='success'
        # assert eval(call_module(module='wordStress', filename='WS_111_toothpaste.wav', sentenceID=111, client=c).data)['status']=='success'

if __name__ == '__main__':
    test_api()