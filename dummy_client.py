import requests
def send_audio(path='audio_recordings/WS_111_toothpaste.wav', url = 'http://localhost:5000/', client=requests):
    with open(path, 'rb') as file:
        files = {'file': file}
        req = client.post(url, files=files)
    return req

def call_module(module='sentenceStress', filename='SS_1_i_would_love_to_go_to_ireland.wav', sentenceID=1, client=requests):
    res = client.post('http://localhost:5000/flowspeech/'+module, data={"sentenceID":str(sentenceID), 'filename':filename})
    print(res.__dict__['_content'])
    return res.__dict__['_content']


def call_vowel_stresses(filename='SS_1_i_would_love_to_go_to_ireland.wav', text='I would love to go to ireland !', client=requests):
    res = client.post('http://localhost:5000/vowel_stresses', data={"text":text, 'filename':filename})
    print(res.__dict__['_content'])
    return res.__dict__['_content']

def call_phoneme_contrast(filename='turned_around.mp3', text='turned around', word_id=0, target='D', alternatives="['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']", client=requests):
    res = client.post('http://localhost:5000/phonemeContrast', data={"text":text, 'filename':filename, 'word_id':word_id, 'alternatives':alternatives, 'target':target})
    print(res.__dict__['_content'])
    return res.__dict__['_content']

if __name__ == "__main__":
    send_audio()
    call_module()
    call_module('hihuhiha')
    call_module(module='wordStress', filename='WS_111_toothpaste.wav', sentenceID=111)

    send_audio(path='audio_recordings/iC_111_slip.wav')

    call_module(sentenceID=111, filename='iC_111_slip.wav', module="iContrast")
    call_vowel_stresses()
    call_vowel_stresses(filename='iC_111_sleep.wav', text='sleep')

    send_audio('audio_recordings/turned_around.mp3')
    call_phoneme_contrast()

    send_audio('audio_recordings/Laaw_MP3.mp3')
    alternatives="['AO0', 'OW0','AO1', 'OW1','AO2', 'OW2']"
    call_phoneme_contrast(filename='Laaw_MP3.mp3', text='law',  word_id=0, target='AO1', alternatives=alternatives)

    send_audio('audio_recordings/iC_112_leave.wav')
    alternatives="['IH0', 'IY0','IH1', 'IY1','IH2', 'IY2']"
    call_phoneme_contrast(filename='iC_112_leave.wav', text='leave',  word_id=0, target='IY1', alternatives=alternatives)
    
    send_audio('audio_recordings/ed_acceptEED.wav')
    # alternatives="['IH0', 'IY0','IH1', 'IY1','IH2', 'IY2']"
    alternatives="['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']"
    # alternatives="['T', 'D', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']"
    # alternatives="['T', 'D']"
    call_phoneme_contrast(filename='ed_acceptEED.wav', text='accepted',  word_id=0, target="IH0 D", alternatives=alternatives)
