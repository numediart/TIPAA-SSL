from speech_tech import *

from text_processing import word_stress_from_text
from module_performance import *
from label_data_processing import *

def test_wordStress():
    d=get_data()

    p=set_params()
    status_audio, rID=prepare_audio_file('audio_recordings/WS_111_toothpaste.wav')
    p['rand_fileName']=rID
    p=make_all_phones_annotation_files(p,remove_special_characters(d[d.analysisId==111].text.values[0]))
    wordStress(p)

    p=set_params()
    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    p=make_all_phones_annotation_files(p,remove_special_characters(d[d.analysisId==1].text.values[0]))
    wordStress(p)

def test_sentenceStress():
    a,textDict=get_sentenceStress_annotation()

    p=set_params()
    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    p=make_all_phones_annotation_files(p,remove_special_characters(textDict[1]))
    sentenceStress(p)

    p=set_params()
    status_audio, rID=prepare_audio_file('audio_recordings/WS_111_toothpaste.wav')
    p['rand_fileName']=rID
    p=make_all_phones_annotation_files(p,remove_special_characters(textDict[1]))
    sentenceStress(p)

def test_audio_formats():
    prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf')
    prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a')
    prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland_stereo.m4a')


# def test_pContrast():
#     p=set_params(sentenceID=11, module="oContrast")
#     # prepare_audio_file(p)
#     status_audio, rID=prepare_audio_file('audio_recordings/Low_WAV.wav')
#     p['rand_fileName']=rID
#     phonemeContrast(p)


# def test_edAnalysis():
#     p=set_params(sentenceID=1, module="edAnalysis")
#     # prepare_audio_file(p)
#     status_audio, rID=prepare_audio_file('audio_recordings/ed_accepted.wav')
#     p['rand_fileName']=rID
#     edAnalysis(p)

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

def test_stress_with_level():
    from text_processing import prefill_for_sentence
    sentence="i would love to go to ireland"

    syllables_data=pd.read_csv('data/syllables.csv')
    d=prefill_for_sentence(sentence, syllables_data)
    d['cmu_phonetics']
    
    p=set_params()
    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    stress_from_formatted_phonetics(d['cmu_phonetics'], level='sentence', p=p)
    
    p=set_params()
    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    stress_from_formatted_phonetics(d['cmu_phonetics'], level='word', p=p)
    
    p=set_params()
    status_audio, rID=prepare_audio_file('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
    p['rand_fileName']=rID
    vowel_stresses_from_phonetics_audio(p=p)


def test_phonemeContrast():
    p=set_params()
    # prepare_audio_file(p)
    status_audio, rID=prepare_audio_file('audio_recordings/turned_around.mp3')
    p['rand_fileName']=rID
    phonemeContrast_from_phonetics_audio(p=p)

    p=set_params()
    status_audio, rID=prepare_audio_file('audio_recordings/turned_around.mp3')
    p['rand_fileName']=rID
    phonemeContrast_from_formatted_phonetics_audio(p=p)

def test_text_processing():
    word_stress_from_text()
    df=generate_prefill_csv()

def test_performance_tests():
    # iContrast_performance_test()
    stress_performance_test()
    stress_performance_test(level='word')
    # wordStress_performance_test()
    # sentenceStress_automatic_annot_performance_test()
    # edAnalysis_performance_test()
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
