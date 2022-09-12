from DL_accuracy_performance import *

def test_pConstrast():
    pContrast_for_actor_recordings(target_phones='AO1')
def test_termination_contrast():
    termination_contrast_from_audiobook_data(data_set='test-other', target_phones='D', n=50)
    termination_contrast_for_actor_recordings()



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

    _, rID=prepare_audio_file(row.audio_file_url)
    n_words_by_chunk=chunk_text(text=row.text)
    res=stress_from_formatted_phonetics(rID,phonetics=row.cmu_phonetics, n_words_by_chunk=n_words_by_chunk, level='word')

    model = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')
    s,fs=librosa.load(row.audio_file_url, sr=16000)
    split_phonetics=sum([p.replace('|','_').split('_') for p in row.cmu_phonetics.split(' ')], [])
    _, p_df, _ = model.align_phones(audio=s,phones=split_phonetics)

    ws = model.compute_stress_score(audio=s,phonetics=row.cmu_phonetics)

def a_test_stress_detection():
    stress_GE_performance_test(level='word')

from utils.text_processing import *
def test_prefill():
    prefill_for_sentence()

    sentence="I paid a $3000 bill when visiting UCLA, it's an expensive hotel, for the 21st century!"
    r=prefill_for_sentence(sentence=sentence)

    assert r['cmu_phonetics']=='AY1 P_EY1_D AH0 {TH_R_IY1 TH_AW1|Z_AH0_N_D D_AA1|L_ER0_Z} B_IH1_L W_EH1_N V_IH1|Z_IH0|T_IH0_NG {Y_UW1 S_IY1 EH1_L EY1} IH1_T_S AE1_N IH0_K_S|P_EH1_N|S_IH0_V HH_OW0|T_EH1_L F_AO1_R DH_AH0 T_W_EH1_N|T_IY0-F_ER1_S_T S_EH1_N|CH_ER0|IY0'

    sentence="A las 22 en punto, tengo una *reunión* con el CEO, Indya, y un ingeniero de una empresa emergente de 30000 dólares en etapa inicial, ¡luego con el CTO!"
    r=prefill_for_sentence(
                        sentence=sentence,
                        syllables_df=pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']), 
                        lang="es_ES",
                        mode='MFA_IPA')  # "CMU" or "MFA_IPA"

    sentence="At 22 o'clock, I have a *meeting* with the CEO, Indya, and an engineer of a 300 k dollars early-stage start-up, then with the CTO!"
    r=prefill_for_sentence(
                        sentence=sentence,
                        # syllables_df=pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']), 
                        lang="en_GB",
                        mode='MFA_IPA')  # "CMU" or "MFA_IPA"
    sentence="A 22 heures, j'ai rendez-vous avec le CEO, Indya, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"
    r=prefill_for_sentence(
                        sentence=sentence,
                        syllables_df=pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']), 
                        lang="fr_FR",
                        mode='MFA_IPA')  # "CMU" or "MFA_IPA"