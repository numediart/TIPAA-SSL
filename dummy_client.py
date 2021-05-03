import requests
def send_audio(path='audio_recordings/WS_111_toothpaste.wav', url = 'http://localhost:5000/'):
    with open(path, 'rb') as file:
        files = {'file': file}
        req = requests.post(url, files=files)

def call_module(module='sentenceStress', filename='SS_1_i_would_love_to_go_to_ireland.wav', sentenceID=1):
    res = requests.post('http://localhost:5000/flowspeech/'+module, {"data[sentenceID]":str(sentenceID), 'data[filename]':filename})
    print(res.__dict__['_content'])