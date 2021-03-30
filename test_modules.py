from speech_tech import *
def test_wordStress():
    r=wordStress()
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='wordStress')
    r=wordStress(p)

def test_sentenceStress():
    sentenceStress()
    sentenceStress(p=set_params(sentenceID=1, waveFileAddress='audio_recordings/WS_111_toothpaste.wav', module='sentenceStress'))

def test_iContrast():
    iContrast()

def test_chunking():
    chunking()

def test_edAnalysis():
    edAnalysis()

from label_data_processing import *

def test_label_data_processing():
    get_wordStress_annotation()
    get_sentenceStress_annotation()
    ed_make_grammars()

from text_processing import word_stress_from_text
def test_text_processing():
    word_stress_from_text()
