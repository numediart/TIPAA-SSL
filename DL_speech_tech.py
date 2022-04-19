from scipy.io.wavfile import  read
import numpy as np
import pandas as pd
from utils.audio_processing import getIntonation, getIntensity, normalize
import soundfile as sf

from utils.text_processing import remove_stress_annots, drop_consecutive_duplicates, drop_consecutive_duplicate_elements, chunk_text, phonetics_indexed_df_from_formatted_phonetics, cmu_vowels, cmu_consonants, unstress, split_phonetics,  cmu_to_gibberish, SonoriPy
from utils.charsiu_utils import charsiu_phone_forced_aligner

# initialize model
model = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')

target_accepted_alternatives={
    'AA': ['AA', 'AO'],
    'AO': ['AA', 'AO'],
    'D': ['D', 'T'],
    'T': ['D', 'T'],
    'IH': ['IH', 'AH', 'EH']
}

def compute_stress_score(textgridData, s, fs):
    """Use textgridData to have the timings of vowels and compute prosody features (intesity, pitch, ...) to compute 
    a value by vowel representing a stress intensity

    Args:
        textgridData ([type]): [description]
        s (np array): audio signal
        fs (int): frequency of sampling
    Returns:
        weighted_score [type]: stress intensity score
    """

    indxVowels=textgridData[textgridData.cmu_phones.isin(vowels)].index.tolist()

    f0Samples=getIntonation(s, fs)
    intensity=getIntensity(s, fs)

    # extract features
    # each word start and end position expressed in samples
    startPositions_samples = (round(fs*textgridData.iloc[:,0])+1).astype(int).tolist()
    stopPositions_samples = round(fs*textgridData.iloc[:,1]).astype(int).tolist()
    
    # to make sure we don t go beyond the end of the signal
    assert stopPositions_samples[-1]<len(s), "The end of the last phoneme should be inside the signal"

    Imax,Imean,Fmax,Fmean,Dur=[],[],[],[],[]
    nVowels=len(indxVowels)
    sylType=np.zeros(nVowels)
    for i in range(nVowels):
        range_vowel=range(startPositions_samples[indxVowels[i]], stopPositions_samples[indxVowels[i]])
        Ivowel=intensity[range_vowel]
        Fvowel=f0Samples[range_vowel]
        
        Imax.append(max(Ivowel))
        Imean.append(np.mean(Ivowel))
        Fmax.append(max(Fvowel))
        Fmean.append(np.mean(Fvowel))

        Dur.append(textgridData['end'].iloc[indxVowels[i]]-textgridData['start'].iloc[indxVowels[i]])
        
        # phone_df=pd.DataFrame([r[2].split('_') for i,r in textgridData.iterrows()])
        # # here we use the prediction of HMM model as an indication, as it has to classify 0, 1 or 2
        # syltype_phone=int(phone_df[2].iloc[indxVowels[i]])
        # if syltype_phone == 2:  # the sylType is 0 for unstressed, 0.5 for secondary stressed syllables and 1 for primary stressed syllables
        #     sylType[i] = 0.5
        # else:
        #     sylType[i]=syltype_phone
    
    # normalization of features (projection to [0 1] range)
    zImax = normalize(Imax)
    zImean = normalize(Imean)
    zFmax = normalize(Fmax)
    zFmean = normalize(Fmean)
    zDur = normalize(Dur)

    # combine the features
    # weighted_score = (zImax + 0.2*zImean + zFmax + 0.2*zFmean + 0.8*zDur + 0.4*sylType)/3.6  # needs fine-tuning once enough user data are available - in the long term train a classifier with annotated user data
    weighted_score = (zImax + 0.2*zImean + zFmax + 0.2*zFmean + 0.8*zDur)/3.2  # needs fine-tuning once enough user data are available - in the long term train a classifier with annotated user data
    return weighted_score

def audio_load_and_check(rID, phonetics, max_speech_rate=8):
    # I first detect if the audio is too short to have a realistic speech rate
    #     https://www.science.org/doi/10.1126/sciadv.aaw2594
    # https://www.reddit.com/r/languagelearning/comments/f5o1om/distribution_of_syllable_rate_sr_in_syllables_per/
    # Speech rate is always between 5 and 8 syl/second
    n_syllables_tot=sum([len(el.split('|')) for el in phonetics.split(' ')])
    try:
        f=sf.SoundFile('./inputs/'+ rID+ '.wav')
    except FileNotFoundError:
        return "error: audio file not found", None
    duration=f.frames / f.samplerate
    speech_rate=n_syllables_tot/duration
    if speech_rate>max_speech_rate: 
        return "success: audio is too short compared to the expected number of syllables", None

    try:
        fs,s=read('./inputs/'+ rID+ '.wav')
    except FileNotFoundError:
        return "error: audio file not found", None
    s=s/32767
    f0Samples=getIntonation(s, fs)
    if sum([el!=el for el in f0Samples])==len(f0Samples):
        return "success: no voiced sound detected (no pitch detected)", None
    return "success", s


def remove_downwards_trend(y):
    if len(y)>2:
        # Remove downwards trend
        x=range(len(y))
        model = np.polyfit(x, y, 1)
        a=model[0]
        b=model[1]
        y=y-(a*x+b)

        # normalize between 0 and 100
        y=y-min(y)
        y=y/max(y)*100
    else:
        y=np.array(y)
    return y.astype(int).tolist()

def intensity_to_bin(score_by_word, n_max=2):
    """
    """
    bin_score_by_word=np.zeros(len(score_by_word)).astype(int).tolist()
    
    if len(score_by_word)==1:
        imax=np.argmax(score_by_word)
        bin_score_by_word[imax]=1
    else:
        imaxes=[np.argmax(score_by_word)] if len(score_by_word)<n_max else np.argpartition(score_by_word, -n_max)[-n_max:]
        for imax in imaxes:
            if score_by_word[imax]>60:
                bin_score_by_word[imax]=1
    return bin_score_by_word

def stress_from_formatted_phonetics(rID,phonetics="AY1 W_UH1_D L_AH1_V T_UW1 G_OW1 T_UW1 AY1|ER0|L_AH0_N_D", 
                                    text="I would love to go to Ireland!", 
                                    level="word", 
                                    chunking_chars=[',',';','.','!','?', ':', '/'],
                                    max_speech_rate=8
                                    ): #'[\,\?\.\!\;\:\"\*]'
    
    status, s = audio_load_and_check(rID, phonetics, max_speech_rate=max_speech_rate)
    if status=="success":
        # print(phonetics)
        ws=model.compute_stress_score(s,phonetics)
        phonetics_indexed_df=phonetics_indexed_df_from_formatted_phonetics(phonetics)
        is_vowel=phonetics_indexed_df.apply(lambda r: unstress(r.phones) in cmu_vowels, axis=1)
        vowels_indexed_df=phonetics_indexed_df[is_vowel]
        # print(vowels_indexed_df)
        vowels_indexed_df.loc[:,'stress_scores']=(100*ws).astype(int)
    else:
        return {"status": status, "stress_intensities": [], "stress_binaries": []}

    word_bins=[]
    word_intensities=[]
    for w_idx in vowels_indexed_df.word_idx.unique():
        word=vowels_indexed_df[vowels_indexed_df.word_idx==w_idx]
        bins=[0]*len(word)
        bins[word.stress_scores.argmax()]=1
        word_bins.append(bins)
        word_intensities.append(word.stress_scores.tolist())

    n_words_by_chunk=chunk_text(text, chunking_chars=chunking_chars)
    max_word_intensities=[max(w) for w in word_intensities]

    scores_grouped_by_chunk=[]
    cumsum=0
    for n in n_words_by_chunk:
        scores_grouped_by_chunk.append(max_word_intensities[cumsum:cumsum+n])
        cumsum+=n
    
    # I tried this on General English data, and in the end, it does not seem to improve
    # scores_grouped_by_chunk=[remove_downwards_trend(el) for el in scores_grouped_by_chunk]

    bins_by_chunk=[]
    for chunk in scores_grouped_by_chunk:
        bin=intensity_to_bin(chunk)
        bins_by_chunk.append(bin)

    if level=="word":
        return {"status": "success", "stress_intensities": word_intensities, "stress_binaries": word_bins}
    elif level=="sentence":
        return {"status": "success", "stress_intensities": sum(scores_grouped_by_chunk,[]), "stress_binaries": sum(bins_by_chunk,[])}
    else:
        return {"status": "error: "+level+"is not a valid level in stress_from_formatted_phonetics. It has to be either 'word' or 'sentence'.", "stress_intensities": [], "stress_binaries": []}


def phonemeContrast_from_formatted_phonetics_audio(rID,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            target_word_idx=0, 
                            target_syllable_idx=0, 
                            target_occurence_idx=0, # will be 0  all the time for vowels, and most of the time for consonants
                            target_phones='ER1',
                            alternatives=cmu_vowels,
                            max_speech_rate=8, **kwargs
                    ):
    phonetics=phonetics.replace('-',' ')
    status, s = audio_load_and_check(rID, phonetics, max_speech_rate=max_speech_rate)
    g_t=[cmu_to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]
    if status=="success":
        phonetic_detection, detected_syllable=model.predict_phone(s, phonetics, target_word_idx, target_syllable_idx, target_phones, target_occurence_idx, phoneme_set=alternatives)

        if detected_syllable!=detected_syllable: 
            return {"status": status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

        g_d=[cmu_to_gibberish[unstress(p)] for p in detected_syllable]
        
        # post-correction: for the target, when we are in a case of accepted alternative in prediction, we replace it with the GT
        if unstress(target_phones) in target_accepted_alternatives:
            if unstress(phonetic_detection) in target_accepted_alternatives[unstress(target_phones)]:
                phonetic_detection=unstress(target_phones)
                g_d=g_t
        
        # phonetic detection needs to be the stressed version for backwards compatibility
        if phonetic_detection==unstress(target_phones): phonetic_detection=target_phones
        return {"status": "success", "phonetic_detection": phonetic_detection, "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}
    else:
        return {"status": status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}
    
def termination_contrast_from_formatted_phonetics_audio(rID,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            target_word_idx=0, 
                            target_phones='D',
                            # termination_basis='[UNK]_D',
                            termination_basis='IH0_D',
                            max_speech_rate=8, **kwargs
                    ):
    phonetics=phonetics.replace('-',' ')
    status, s = audio_load_and_check(rID, phonetics, max_speech_rate=max_speech_rate)

    phonetics_indexed_df=phonetics_indexed_df_from_formatted_phonetics(phonetics.split(' ')[target_word_idx])
    idx_syl_ter=phonetics_indexed_df.syl_idx.iloc[-1]
    df_syl_GT=phonetics_indexed_df[phonetics_indexed_df.syl_idx==idx_syl_ter]
    syl_GT=remove_stress_annots(df_syl_GT.phones.tolist())
    g_t=[cmu_to_gibberish[unstress(p)] for p in syl_GT]
    # g_t=[cmu_to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]
    if status=="success":

        n_p_target=len(target_phones.split('_'))
        split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
        target_word=split_phonetics[target_word_idx]
        split_phonetics[target_word_idx]=target_word[:-n_p_target]+termination_basis.split('_')
        
        # same change for indexes
        syl_idxs=phonetics_indexed_df.syl_idx.tolist()
        syl_idxs=syl_idxs[:-n_p_target]+[syl_idxs[-1]]*len(termination_basis.split('_'))

        df_word=model.predict_word(s, split_phonetics, target_word_idx)
        df_word['syl_idx']=syl_idxs
        df_syl=df_word[df_word.syl_idx==syl_idxs[-1]]

        pred_phones=df_syl[df_syl.pred_phones_audio!='[SIL]']
        syl_detected=drop_consecutive_duplicates(pred_phones[['pred_phones_audio']]).pred_phones_audio.tolist()

        # root based on GT
        root=syl_GT[:-n_p_target-1]
        # detected termination
        ter=syl_detected[-n_p_target-1:]
        GT=syl_GT[-n_p_target-1:]
        
        if len(ter)==len(GT):
            ter_post=[]
            for i,p in enumerate(ter):
                # post-correction: for the target, when we are in a case of accepted alternative in prediction, we replace it with the GT
                if unstress(GT[i]) in target_accepted_alternatives:
                    if unstress(p) in target_accepted_alternatives[unstress(GT[i])]:
                        ter_post.append(unstress(GT[i]))
                    else:
                        ter_post.append(unstress(p))
                else:
                    ter_post.append(unstress(p))
        else: ter_post=ter

        # there might be consecutive duplicates when we concatenate root and ter_post
        g_d=drop_consecutive_duplicate_elements([cmu_to_gibberish[unstress(p)] for p in root+ter_post])
        
        # phonetic detection needs to be the stressed version for backwards compatibility 
        # (however, here I convert to stressed version only when correct, it might work, but could cause problems?)
        detection=ter_post[1:]
        if target_phones!='': # case of final -s
            if detection==remove_stress_annots(target_phones.split('_')): detection=target_phones.split('_')

        return {"status": "success", "phonetic_detection": '_'.join(detection), "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}
    else:
        return {"status": status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

def phonetic_content_analysis(s, phonetics):
    phonetic_content=model.analyze_phonetic_content(s, phonetics)
    if len(phonetic_content)==0: return phonetic_content
    
    
    def syl_analysis(syl_df):
        # inside a syllable or word, there cannot be several times the same phoneme consecutively
        collapsed_syl=drop_consecutive_duplicates(syl_df[['pred_phones_audio']])

        # one syllable in ground truth can correspond in several syllables in prediction, e.g. moved -> movED
        # or also in "0 syllable" if there is no vowel. If that's the case,  I have to consider it is 1 syllable
        pred_syls=SonoriPy(collapsed_syl.pred_phones_audio.tolist())[0]
        if pred_syls==[]: pred_syls=[collapsed_syl.pred_phones_audio.tolist()]

        
        # extract syllable indices for predicted syls
        pred_syls_indxs=sum([[i]*n for i,n in enumerate([len(syl) for syl in pred_syls])], [])

        collapsed_syl.loc[:,'pred_syls_indxs_inside_GT_syl']=pred_syls_indxs

        # I align the collapsed syllable to the timed one. This leads to NaNs that have to be filled
        syl_df.loc[:,'pred_syls_indxs_inside_GT_syl']=collapsed_syl.loc[:,'pred_syls_indxs_inside_GT_syl'].astype(int)
        syl_df=syl_df.fillna(method="ffill")

        # in each syllable in prediction, I only keep one vowel, by majority vote, i.e. I drop all vowels except max frames in each syl
        for i in syl_df.pred_syls_indxs_inside_GT_syl.unique():
            s=syl_df.loc[syl_df.pred_syls_indxs_inside_GT_syl==i]
            v=s[s.pred_phones_audio.isin(cmu_vowels)]
            if len(v)>0:
                m=v.n_frames.idxmax()
                l=[el for el in v.index.tolist() if el != m]
                syl_df=syl_df.drop(l)
        
        return syl_df

    # reduction: we go in each syllable
    dfs=[]
    for w_idx in range(phonetic_content.word_idx.values[-1]+1):
        w_df=phonetic_content[phonetic_content.word_idx==w_idx]
        for s_idx in range(w_df.syl_idx.values[-1]+1):
            s_df=w_df[phonetic_content.syl_idx==s_idx]
            syl_df=syl_analysis(s_df)
            dfs.append(syl_df)
    phonetic_content=pd.concat(dfs)

    # post-correction : for each row when we are in a case of accepted alternative in prediction, we replace it with the GT
    for i,r in phonetic_content.iterrows():
        if r.pred_phones in target_accepted_alternatives:
            if r.pred_phones_audio in target_accepted_alternatives[r.pred_phones]:
                phonetic_content.loc[i, 'pred_phones_audio']=r.pred_phones

    
    # phonetic_content=phonetic_content[phonetic_content.n_frames>1]
    phonetic_content=phonetic_content[phonetic_content.pred_phones_audio!='[SIL]']

    # phonetic_content.loc[phonetic_content.pred_phones_audio=='[SIL]','pred_phones_audio']=''

    phonetic_content=phonetic_content.loc[drop_consecutive_duplicates(phonetic_content[['pred_phones','pred_phones_audio']]).index,:]
    return phonetic_content

def syllable_contrast_from_formatted_phonetics_audio(rID,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            target_word_idx=0, 
                            target_syllable_idx=0,
                            max_speech_rate=8
                    ):
    phonetics=phonetics.replace('-',' ')
    status, s = audio_load_and_check(rID, phonetics, max_speech_rate=max_speech_rate)
    g_t=[cmu_to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]

    if status=="success":
        phonetic_content=phonetic_content_analysis(s, phonetics)
        if len(phonetic_content)==0: return {"status": "phonetic_content is empty" ,  "gibberish_truth":  "null", "gibberish_detected":  "null"}

        syllable_content=phonetic_content[phonetic_content.word_idx==target_word_idx][phonetic_content.syl_idx==target_syllable_idx]
        detected_syllable=syllable_content.pred_phones_audio.tolist()
        g_d=drop_consecutive_duplicate_elements([cmu_to_gibberish[unstress(p)] for p in detected_syllable])

        return {"status": "success", "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}
    else:
        return {"status": status,  "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}
    
