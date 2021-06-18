import requests
import json
from text_processing import phonetics_from_sentence
def send_audio(path='audio_recordings/WS_111_toothpaste.wav', url = 'http://localhost:5000/upload', client=requests):
    with open(path, 'rb') as file:
        files = {'file': file}
        req = client.post(url, files=files)
    return req

def call_vowel_stresses(rID, text='I would love to go to ireland !', client=requests):
    phonetics=phonetics_from_sentence(text)
    res = client.post('http://localhost:5000/vowel_stresses', data={"phonetics":json.dumps(phonetics), 'rID':rID})
    print(res.__dict__['_content'])
    return res.__dict__['_content']


def call_module(rID, text='I would love to go to ireland !', module='sentenceStress', client=requests):
    phonetics=phonetics_from_sentence(text)
    res = client.post('http://localhost:5000/flowspeech/'+module, data={"phonetics":json.dumps(phonetics), 'rID':rID})
    print(res.__dict__['_content'])
    return res.__dict__['_content']

def call_phoneme_contrast(rID, text='turned around', word_id=0, target='D', alternatives="['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']", client=requests):
    phonetics=phonetics_from_sentence(text)
    res = client.post('http://localhost:5000/phonemeContrast', data={"phonetics":json.dumps(phonetics), 'rID':rID, 'word_id':word_id, 'alternatives':alternatives, 'target':target})
    print(res.__dict__['_content'])
    return res.__dict__['_content']


# deprecated functions
if False:
    def call_module(module='sentenceStress', filename='SS_1_i_would_love_to_go_to_ireland.wav', sentenceID=1, client=requests):
        res = client.post('http://localhost:5000/flowspeech/'+module, data={"sentenceID":str(sentenceID), 'filename':filename})
        print(res.__dict__['_content'])
        return res.__dict__['_content']



if __name__ == "__main__":
    res=send_audio(path='audio_recordings/turned_around.mp3')
    rID=res.__dict__['_content']
    call_phoneme_contrast( rID.decode('utf-8'))

    
    res=send_audio(path='audio_recordings/I_visited_italy.mp3')
    rID=res.__dict__['_content']
    call_phoneme_contrast(rID.decode('utf-8'), text='I visited italy', word_id=1, target='IH0', alternatives="['IH0', 'IY0']")

    
    res=send_audio(path='audio_recordings/I_visited_italy.mp3')
    rID=res.__dict__['_content']
    call_phoneme_contrast(rID.decode('utf-8'), text='I visited italy', word_id=1, target='IH0 D', alternatives="['IH0 D', 'D', 'T']")


    res=send_audio(path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    rID=res.__dict__['_content']
    call_vowel_stresses(rID.decode('utf-8'))

    
    res=send_audio(path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    rID=res.__dict__['_content']
    call_module(rID.decode('utf-8'))

    
    res=send_audio(path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    rID=res.__dict__['_content']
    call_module(rID.decode('utf-8'), module="wordStress")

    # res=send_audio(path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    # rID=res.__dict__['_content']
    # call_sentenceStress(rID.decode('utf-8'))

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
