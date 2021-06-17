from speech_tech import *

from text_processing import word_stress_from_text
from module_performance import *
from label_data_processing import *

def test_wordStress():
    p=set_params(sentenceID=111, module='wordStress')
    status_audio, rID=prepare_audio_file('audio_recordings/WS_111_toothpaste.wav')
    p['rand_fileName']=rID
    wordStress(p)

    p=set_params(sentenceID=1, module='wordStress')
    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    wordStress(p)

def test_sentenceStress():
    a,textDict=get_sentenceStress_annotation()

    p=set_params(sentenceID=1, module='sentenceStress')
    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    p=make_all_phones_annotation_files(p,remove_special_characters(textDict[p['sentenceID']]))
    sentenceStress(p)

    p=set_params(sentenceID=1, module='sentenceStress')
    status_audio, rID=prepare_audio_file('audio_recordings/WS_111_toothpaste.wav')
    p['rand_fileName']=rID
    p=make_all_phones_annotation_files(p,remove_special_characters(textDict[p['sentenceID']]))
    sentenceStress(p)

def test_audio_formats():
    prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf')
    prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a')
    prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland_stereo.m4a')


def test_pContrast():
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Low_WAV.wav', module="oContrast")
    # prepare_audio_file(p)
    status_audio, rID=prepare_audio_file('audio_recordings/Low_WAV.wav')
    p['rand_fileName']=rID
    phonemeContrast(p)


def test_edAnalysis():
    p=set_params(sentenceID=1, waveFileAddress='audio_recordings/ed_accepted.wav', module="edAnalysis")
    # prepare_audio_file(p)
    status_audio, rID=prepare_audio_file('audio_recordings/ed_accepted.wav')
    p['rand_fileName']=rID
    edAnalysis(p)

def test_vowel_stresses():
    p=set_params()
    # prepare_audio_file(p)
    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    vowel_stresses_from_phonetics_audio(p=p)

    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    sentenceStress_from_phonetics_audio(p=p)

    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    wordStress_from_phonetics_audio(p=p)



def test_phonemeContrast():
    p=set_params(waveFileAddress='audio_recordings/turned_around.mp3')
    # prepare_audio_file(p)
    status_audio, rID=prepare_audio_file('audio_recordings/turned_around.mp3')
    p['rand_fileName']=rID
    phonemeContrast_from_phonetics_audio(p=p)


def test_text_processing():
    word_stress_from_text()

def test_performance_tests():
    # iContrast_performance_test()
    sentenceStress_performance_test()
    wordStress_performance_test()
    # sentenceStress_automatic_annot_performance_test()
    edAnalysis_performance_test()
    edAnalysis_from_audiobook_data(data_set='dev-clean', n=100)
    pContrast_from_audiobook_data(n=100)

def test_label_data_processing():
    get_wordStress_annotation()
    get_sentenceStress_annotation()
    # ed_make_grammars()
    # get_data()
    make_dct_all_phones_from_text()
    make_generic_dct_from_phonetics()
    make_grammar_from_dct()
    make_all_phones_annotation_files()
    make_all_phones_annotation_files_from_phonetics()
    make_pContrast_annotation_files_from_phonetics()
    make_pContrast_annotation_files()
