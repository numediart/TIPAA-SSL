import requests
def send_audio(path='audio_recordings/WS_111_toothpaste.wav', url = 'http://localhost:5000/'):
    with open(path, 'rb') as file:
        files = {'file': file}
        req = requests.post(url, files=files)

def call_module(module='sentenceStress', filename='SS_1_i_would_love_to_go_to_ireland.wav', sentenceID=1):
    res = requests.post('http://localhost:5000/flowspeech/'+module, {"sentenceID":str(sentenceID), 'filename':filename})
    print(res.__dict__['_content'])

def call_vowel_stresses(filename='SS_1_i_would_love_to_go_to_ireland.wav', text='I would love to go to ireland !'):
    res = requests.post('http://localhost:5000/vowel_stresses/', {"text":text, 'filename':filename})
    print(res.__dict__['_content'])

def call_phoneme_contrast(filename='turned_around.mp3', text='turned around', word_id=0):
    res = requests.post('http://localhost:5000/phonemeContrast/', {"text":text, 'filename':filename, 'word_id':word_id})
    print(res.__dict__['_content'])