from flask_server import app
from dummy_client import *
import os
os.environ['FLOWSPEECH_KEY']="ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"
import ast


def test_api():
    # c=app.test_client()
    with app.test_client() as c:
        res=send_audio_base64(client=c)
        assert res.status_code == 200

        res=ast.literal_eval(res.data.decode('utf-8'))
        assert res['status']=='success'
        rID=res['rID']
        print(res)
        res=call_phoneme_contrast(rID, client=c)
        assert res.status_code == 200

        base_url = 'http://localhost:8000'
        crash_test(base_url=base_url, client=c)

        # assert c.get('http://localhost:8000/').status_code == 200
        # assert send_audio(client=c).status_code==200
        # assert eval(call_module(client=c).data)['status']=='success'
        # assert eval(call_phoneme_contrast(client=c).data)['status']=='success'
        # assert eval(call_module(module='wordStress', filename='WS_111_toothpaste.wav', sentenceID=111, client=c).data)['status']=='success'

if __name__ == '__main__':
    test_api()