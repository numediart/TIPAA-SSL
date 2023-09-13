from performance_functions import pContrast_on_synth_words, final_ed_from_audiobook_data, final_s_from_audiobook_data, stress_GE_performance_test
from DL_speech_tech import default_model, phone_prob_matrix_segmentation, stress_from_formatted_phonetics, phonemeContrast_from_formatted_phonetics_audio, start_end_contrast_from_formatted_phonetics_audio, compute_stress_score, multiple_aspect_from_formatted_phonetics_audio

from src.audio_processing import prepare_audio_file, read_audio_file
from src.text_processing import prefill_for_sentence, get_augmented_mfa_dict, chunk_text, remove_stress_annots
from src.label_data_processing import actor_recordings
import base64
from src.pronunciation_dictionaries import cmu_vowels, cmu_consonants
from linetimer import CodeTimer

import pandas as pd

def a_test_pConstrast():
    # pContrast_for_actor_recordings(target_phones='AO1', n=10)
    # phoneme_confusions(n=10, performance_function=pContrast_on_synth_words)
    n=10
    p='EY1'
    preds, rates=pContrast_on_synth_words(target_phones=p, n=n, alternatives=cmu_vowels, accent='UK')
    
def a_test_termination_contrast():
    final_ed_from_audiobook_data(data_set='test-other', target_phones='D', n=20)
    # termination_contrast_for_actor_recordings()
    final_s_from_audiobook_data(n=100)
    

def test_DL_speech_tech_functions():
    from src.label_data_processing import actor_recordings

    df=actor_recordings()

    path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf'
    # path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a'
    # path='data/audio_recordings/turned_around.mp3'
    encode_string = base64.b64encode(open(path, "rb").read())
    formatted_phonetics=prefill_for_sentence('I would love to go to ireland')['phonetics']
    res=stress_from_formatted_phonetics(encode_string,phonetics=formatted_phonetics, 
                                    level="sentence", 
                                    n_words_by_chunk=[7],
                                    max_speech_rate=8, mode='base64'
                                    )
    assert res['stress_binaries'][2]==1, "Stress detection failed"

    
    # path='data/During the nineteen sixties Gregory became active in civil rights.wav'
    # encode_string = base64.b64encode(open(path, "rb").read())
    # formatted_phonetics=prefill_for_sentence('During the nineteen sixties Gregory became active in civil rights')['phonetics']

    detection_df=multiple_aspect_from_formatted_phonetics_audio(encode_string,
                                    phonetics=formatted_phonetics, 
                                    max_speech_rate=8, mode='base64'
                                    )
    assert sum(detection_df.phones!=detection_df.detection)/len(detection_df) < 0.2, "Test example has a too high phoneme error rate"
    
    # try with nonsense phonetics
    # df.text[df.text.str.split(' ').apply(len)==1]

    row=df[df.text=='it'].iloc[0]
    s,fs=read_audio_file(row.audio_file_url, fs=16000)
    formatted_phonetics=row.cmu_phonetics
    detection_df=multiple_aspect_from_formatted_phonetics_audio(s,
                                    phonetics=formatted_phonetics, 
                                    max_speech_rate=8, mode='numpy'
                                    )
    assert sum(detection_df.phones!=detection_df.detection)/len(detection_df) == 0, "Test example has a too high phoneme error rate.The audio contains a native pronunciation of 'IH1_T'"

    df[df.text=='One *hundred* percent.'].text
    row=df[df.text=='One *hundred* percent.'].iloc[0]
    s,fs=read_audio_file(row.audio_file_url, fs=16000)

    formatted_phonetics=row.cmu_phonetics
    res=stress_from_formatted_phonetics(s,phonetics=formatted_phonetics, 
                                    level="sentence", 
                                    n_words_by_chunk=[3],
                                    max_speech_rate=8, mode='numpy'
                                    )
    assert res['stress_binaries'][1]==1, "Stress detection failed"
    res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics, 
                            target_word_idx=1,
                            target_phones='D',
                            basis='IH0_D', mode="numpy"
                    )
    assert res['phonetic_detection']=="IH_D", "final -ed detection failed"
    res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=row.cmu_phonetics, 
                            target_word_idx=1,
                            target_syllable_idx=0,
                            target_phones='HH',
                            basis='HH', position="start", mode="numpy"
                    )
    assert res['phonetic_detection']=="HH", "/h/ sound detection failed"


def test_particular_cases():
    df=actor_recordings()
    len_t_seg=df.apply(lambda r: len(r.syllable_parts.split(' ')), axis=1)
    len_p=df.apply(lambda r: len(r.cmu_phonetics.split(' ')), axis=1)
    df[len_t_seg!=len_p]

    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    row=df_word_stress[df_word_stress.text.str.contains('grandma')].iloc[-1]

    # row=df_sentence_stress.iloc[0]

    s,fs=read_audio_file(row.audio_file_url, fs=16000)
    n_words_by_chunk=chunk_text(text=row.text)
    res=stress_from_formatted_phonetics(s,phonetics=row.cmu_phonetics, n_words_by_chunk=n_words_by_chunk, level='word', mode="numpy")

    # model = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')
    s,fs=read_audio_file(row.audio_file_url, fs=16000)
    split_phonetics=sum([p.replace('|','_').split('_') for p in row.cmu_phonetics.split(' ')], [])
    
    split_phonetics=[p.replace('|','_').split('_') for p in row.cmu_phonetics.split(' ')]
    split_phonetics=sum(split_phonetics,[])
    with CodeTimer('whole phone prediction'):
        # df_segmented = default_model.predict_with_timings(s, remove_stress_annots(split_phonetics))
        phone_prob_matrix = default_model.predict_phone_prob_matrix(s, default_model.fs)    
        df_segmented=phone_prob_matrix_segmentation(phone_prob_matrix, remove_stress_annots(split_phonetics), model=default_model)

    # with CodeTimer('stress extraction'): ws=model.compute_stress_score(s,phonetics)
    with CodeTimer('stress extraction'): ws=compute_stress_score(df_segmented, s, fs=default_model.fs)

def a_test_stress_detection():
    stress_GE_performance_test(level='word')

def test_prefill():
    prefill_for_sentence()

    sentence="I paid a $3000 bill when visiting UCLA, it's an expensive hotel, for the 21st century!"
    r=prefill_for_sentence(sentence=sentence)

    assert r['phonetics']=='AY1 P_EY1_D AH0 {TH_R_IY1 TH_AW1|Z_AH0_N_D D_AA1|L_ER0_Z} B_IH1_L W_EH1_N V_IH1|Z_IH0|T_IH0_NG {Y_UW1 S_IY1 EH1_L EY1} IH1_T_S AE1_N IH0_K_S|P_EH1_N|S_IH0_V HH_OW0|T_EH1_L F_AO1_R DH_AH0 T_W_EH1_N|T_IY0-F_ER1_S_T S_EH1_N|CH_ER0|IY0'

    sentence="A las 22 en punto, tengo una *reunión* con el CEO, Indya, y un ingeniero de una empresa emergente de 30000 dólares en etapa inicial, ¡luego con el CTO!"
    r=prefill_for_sentence(
                        sentence=sentence,
                        syllables_df=pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']), 
                        lang="es_ES",
                        mode='MFA_IPA',  word_dict=get_augmented_mfa_dict("es_ES"))  # "CMU" or "MFA_IPA"

    sentence="At 22 o'clock, I have a *meeting* with the CEO, Indya, and an engineer of a 300 k dollars early-stage start-up, then with the CTO!"
    r=prefill_for_sentence(
                        sentence=sentence,
                        # syllables_df=pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']), 
                        lang="en_GB",
                        mode='MFA_IPA',  word_dict=get_augmented_mfa_dict("en_GB"))  # "CMU" or "MFA_IPA"
    
    sentence="A 22 heures, j'ai rendez-vous avec la CEO, Indya, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"
    r=prefill_for_sentence(
                        sentence=sentence,
                        syllables_df=pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']), 
                        lang="fr_FR",
                        mode='MFA_IPA',  word_dict=get_augmented_mfa_dict("fr_FR"))  # "CMU" or "MFA_IPA"

