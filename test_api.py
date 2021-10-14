from flask_server import app
from dummy_client import *
import os
os.environ['FLOWSPEECH_KEY']="ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"
import ast


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

if __name__ == '__main__':
    test_api()