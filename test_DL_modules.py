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