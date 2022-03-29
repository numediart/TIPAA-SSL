from scipy.io.wavfile import write, read
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from utils.audio_processing import load_audio, getIntonation, getIntensity, normalize, getf0Samples, prepare_audio_file
import soundfile as sf
from utils.htk_utils import get_textgrid_data, clean_htk_files

from utils.label_data_processing import make_all_phones_annotation_files_from_phonetics, make_pContrast_annotation_files_from_phonetics
from utils.text_processing import phonetics_from_sentence, chunk_text, phonetics_indexed_df_from_formatted_phonetics, cmu_vowels, unstress
import uuid
import time

from utils.charsiu_utils import charsiu_phone_forced_aligner

# initialize model
model = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')


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
                                    chunking_chars=[',',';','.','!','?', ':'],
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
    
    # I tried thisx on General English data, and in the end, it does not seem to improve
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
        print("No such level in stress_from_formatted_phonetics. It has to be either 'word' or 'sentence'.")
        return {"status": "error: "+level+"is not a valid level in stress_from_formatted_phonetics. It has to be either 'word' or 'sentence'.", "stress_intensities": [], "stress_binaries": []}


def phonemeContrast_from_formatted_phonetics_audio(rID,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            # p=set_params(), 
                            target_word_idx=0, 
                            target_syllable_idx=0, 
                            # alternatives=['T', 'D', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']
                            # alternatives='T D IH0_D IH1_D IH2_D EH2_D AH0_D',
                            target_phones='D',
                            max_speech_rate=8
                    ):
    status, s = audio_load_and_check(rID, phonetics, max_speech_rate=max_speech_rate)

    if status=="success":
        phonetic_detection=model.predict_phone(s, phonetics, target_word_idx, target_syllable_idx, target_phones)
        return {"status": "success", "phonetic_detection": phonetic_detection}
    else:
        return {"status": status, "phonetic_detection": float('nan')}
    
    
