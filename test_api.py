from flask_server import app
# from dummy_client import *
import requests
# c=app.test_client()

def call_module(module='sentenceStress', filename='SS_1_i_would_love_to_go_to_ireland.wav', sentenceID=1, client=requests):
    res = client.post('http://localhost:5000/flowspeech/'+module, data={"sentenceID":str(sentenceID), 'filename':filename})
    print(res)
    return res

def call_phoneme_contrast(filename='turned_around.mp3', text='turned around', word_id=0, target='D', alternatives="['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']", client=requests):
    res = client.post('http://localhost:5000/phonemeContrast', data={"text":text, 'filename':filename, 'word_id':word_id, 'alternatives':alternatives, 'target':target})
    print(res)
    return res

def test_api():
    with app.test_client() as c:
        assert c.get('/').status_code == 200
        # assert send_audio(client=c).status_code==200
        assert eval(call_module(client=c).data)['status']=='success'
        assert eval(call_phoneme_contrast(client=c).data)['status']=='success'
        assert eval(call_module(module='wordStress', filename='WS_111_toothpaste.wav', sentenceID=111, client=c).data)['status']=='success'