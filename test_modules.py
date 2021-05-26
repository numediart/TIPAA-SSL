from speech_tech import *
def test_wordStress():
    r=wordStress()
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='wordStress')
    r=wordStress(p)

def test_sentenceStress():
    sentenceStress()
    sentenceStress(p=set_params(sentenceID=1, waveFileAddress='audio_recordings/WS_111_toothpaste.wav', module='sentenceStress'))

def test_audio_formats():
    sentenceStress(p=set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf', module='sentenceStress'))
    sentenceStress(p=set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a', module='sentenceStress'))
    sentenceStress(p=set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland_stereo.m4a', module='sentenceStress'))

def test_iContrast():
    iContrast()

def test_pContrast():
    phonemeContrast()

def test_chunking():
    chunking()

def test_edAnalysis():
    edAnalysis()

from label_data_processing import *
def test_label_data_processing():
    get_wordStress_annotation()
    get_sentenceStress_annotation()
    # ed_make_grammars()

from text_processing import word_stress_from_text
def test_text_processing():
    word_stress_from_text()

from module_performance import *
def test_performance_tests():
    iContrast_performance_test()
    sentenceStress_performance_test()

    wordStress_performance_test()
    sentenceStress_automatic_annot_performance_test()
    edAnalysis_performance_test()
    edAnalysis_from_audiobook_data(data_set='dev-clean', n=100)
    pContrast_from_audiobook_data(n=100)

from label_data_processing import *
def test_label_data_processing():
    # get_data()
    make_generic_dct_from_text()
    make_generic_dct_from_phonetics()
    make_grammar_from_dct()
